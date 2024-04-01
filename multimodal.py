import torch
import torch.nn as nn
from torch.nn import LSTM
import torch.optim as optim
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import pandas as pd
import numpy as np

class MultimodalDataset(Dataset):
    '''
    Parameters
    ----------
    static: the static DataFrame
    dynamic: dynamic dataset where each timestep is concatenated into one dimension
        ```
            id
            20008098    [[21.0, 20.0, 21.0, 8.9, 100.0, 0.9, 178.0, 13...
            20013244    [[13.0, 29.0, 17.0, 8.9, 103.0, 1.1, 127.0, 14...
            20015730    [[11.0, 25.0, 17.0, 8.1, 112.0, 1.6, 121.0, 14...
            20020562    [[12.0, 22.0, 21.0, 8.2, 104.0, 2.2, 91.0, 134...
            20021110    [[16.0, 27.0, 32.0, 9.7, 103.0, 1.2, 88.0, 141...
        ```

    id_lengths: a dictionary where the key is the patient_id and the value is the true length the time series associated with each patient id (to be used for packed padding)
        ```
            {
                20008098: 9,
                20013244: 7,
                20015730: 10,
                20020562: 10,
                20021110: 10,
                20022095: 6,
                20022465: 6,
                20024177: 7
            }

    notes: the notes DataFrame

    Outputs
    -------
    packed_dynamic_X: A sequence of time steps, dynamically packed and padded, representing data for a specific patient
    notes_X: A sequence of tokenized clinical notes for the same patient
    notes_intervals: an array where each index i dictates the difference between the last timestep and the timestep at index i in days.
        ```
            array([3, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0]), 
            where the interval at index 0 indicates that the note at this index was written 3 days before the last entry
    '''

    def __init__(self, static, dynamic, id_lengths, notes):
        self.static = static
        self.static_dict = static.set_index('id')['los_icu'].to_dict()
        self.dynamic = dynamic
        self.id_lengths = id_lengths
        self.notes = notes
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    def __len__(self):
        return len(self.static)
    
    def __getitem__(self, idx):
        patient_id = self.static.iloc[idx]['id']

        # time series
        dynamic_X = self.dynamic[patient_id]
        dynamic_X = torch.tensor(dynamic_X, dtype=torch.float32)
        patient_timesteps = self.id_lengths[patient_id]

        # notes
        notes = self.notes[self.notes['id'] == patient_id]['text'].tolist()
        notes_intervals = self.notes[self.notes['id'] == patient_id]['interval'].to_numpy()
        notes_intervals = torch.tensor(notes_intervals, dtype=torch.float32)
        notes_X = self.tokenizer(notes, return_tensors='pt', truncation=True, max_length=512, padding='max_length')

        # los
        los = [self.static_dict.get(patient_id, [])]
        return dynamic_X, patient_timesteps, notes_X, notes_intervals, los
    
class MultimodalNetwork(nn.Module):
    '''
    time_series_model: expects an input of packed padded sequences
    text_model: expects an input of dict with keys {'input_ids', 'token_type_ids', 'attention_mask'} of tokenized sequences
    '''
    def __init__(self, input_size, out_features, hidden_size, decay_factor=0.9, batch_first=True, **kwargs):
        super(MultimodalNetwork, self).__init__(**kwargs)
        self.decay_factor = decay_factor
        
        self.time_series_model = LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first)
        self.text_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        self.fc = nn.Sequential(
            nn.LayerNorm(normalized_shape=hidden_size + 768),
            nn.Linear(in_features=hidden_size + 768, out_features=out_features, bias=True),
            nn.Softplus()
        )

    def weighted_sum(self, embeddings, interval, decay_factor):
        device = embeddings.device
        weights = (decay_factor ** interval).to(device)
        weighted_sum = torch.matmul(weights, embeddings)

        return weighted_sum

    def forward(self, packed_dynamic_X, notes_X_batch, notes_intervals_batch):
        _, (ht, _) = self.time_series_model(packed_dynamic_X)
        ht = ht[-1]

        embeddings = []
        for (patient_notes, notes_interval) in zip(notes_X_batch, notes_intervals_batch):
            patient_embeddings = self.text_model(**patient_notes).pooler_output
            weighted_sum = self.weighted_sum(embeddings=patient_embeddings, interval=notes_interval, decay_factor=self.decay_factor)
            embeddings.append(weighted_sum)

        zt = torch.stack(embeddings)
        combined_representation = torch.cat((ht, zt), dim=1)
        
        y_pred = self.fc(combined_representation)

        return y_pred
    
def collation(batch):
    dynamic_X, patient_timesteps, notes_X_batch, notes_intervals_batch, los_batch = zip(*batch)
    padded_dynamic = pad_sequence(dynamic_X, batch_first=True, padding_value=0.0)
    packed_dynamic_X = pack_padded_sequence(input=padded_dynamic, lengths=patient_timesteps, batch_first=True, enforce_sorted=False)

    los_batch= torch.tensor(los_batch, dtype=torch.float32)

    return packed_dynamic_X, notes_X_batch, notes_intervals_batch, los_batch