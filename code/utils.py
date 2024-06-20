# This file contains helper functions for Multi-label Topic Classification by fine-tuning a BERT model
# The code has been adapted from: [https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb]
# The functions are used in BERT_1step.ipynb and BERT_2step.ipynb

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import transformers
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix

class MultiLabelDataset(Dataset):
    r"""PyTorch Dataset for handling multilabel text data.

    This class handles the preprocessing and formatting of text data for multilabel classification tasks, where each text can have multiple labels.
    Texts are tokenized and encoded, and transformed into a suitable format for training with a BERT-based model.

    Arguments:
        dataframe (:obj:`pandas.DataFrame`):
            The dataframe containing the texts and their corresponding labels.
        tokenizer (:obj:`transformers.PreTrainedTokenizer`):
            The tokenizer used to convert text into tokens and token IDs.
        max_len (:obj:`int`):
            The maximum length of the tokenized output. Texts will be truncated or padded to this length.

    Attributes:
        text (:obj:`pandas.Series`):
            Series containing all texts from the dataframe.
        targets (:obj:`pandas.Series`):
            Series containing all label lists associated with the texts.
        max_len (:obj:`int`):
            Maximum token length.
    """
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets =self.data.labels
        self.max_len = max_len

    def __len__(self):
        r"""Returns the number of examples in the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        r"""Retrieves an item in tokenized format ready for model consumption.

        This method formats text data by tokenizing and converting it into tensors, which include token IDs, attention masks, and token type IDs, and their corresponding targets.

        Arguments:
            index (:obj:`int`):
                Index position of the data point to retrieve.

        Returns:
            :obj:`Dict[str, torch.Tensor]`: A dictionary containing processed
            text data and labels, ready to be fed into a BERT model.
        """
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'text': text,
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    
class BERTClass(torch.nn.Module):
    r"""BERT Model for Multilabel Classification.

    This model builds on the BERT architecture and extends it with a custom linear layer for multilabel classification. It is designed to process
    batches of tokenized text, applying dropout regularization and producing predictions for multiple labels simultaneously.

    Attributes:
        l1 (:obj:`transformers.BertModel`):
            The BERT model pre-loaded with 'bert-base-uncased' configuration.
        l2 (:obj:`torch.nn.Dropout`):
            Dropout layer for regularization.
        l3 (:obj:`torch.nn.Linear`):
            Linear layer to produce predictions for each label from BERT's output.
    """
    def __init__(self, NUM_LABELS):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.1)
        self.l3 = torch.nn.Linear(768, NUM_LABELS)

    def forward(self, ids, mask, token_type_ids):
        r"""Forward pass of the model.

        Takes token IDs, masks, and token type IDs, and processes them through BERT, a dropout layer, and a linear layer to produce final predictions.

        Arguments:
            ids (:obj:`torch.Tensor`):
                The tensor of input token IDs.
            mask (:obj:`torch.Tensor`):
                The attention mask tensor.
            token_type_ids (:obj:`torch.Tensor`):
                Token type IDs.

        Returns:
            :obj:`torch.Tensor`: The output from the linear layer, representing
            predicted logits for each label.
        """
        _, output_1= self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

def loss_fn(outputs, targets):
    r"""Calculates the loss using BCEWithLogitsLoss.
    This function is used during training to compute the binary cross-entropy loss between the predicted outputs and the actual targets for multilabel classification.

    Arguments:
        outputs (:obj:`torch.Tensor`):
            The logits predicted by the model.
        targets (:obj:`torch.Tensor`):
            The actual labels for the data points.

    Returns:
        :obj:`torch.Tensor`: The computed loss as a tensor.
    """
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def train(model, device, training_loader, epoch, optimizer, scheduler):
    r"""Trains the model for the defined number of epochs through all batches in the training loader.

    This function processes each batch from the training loader, computes the loss, performs a backward pass, and updates the model's weights. 
    It also handles the scheduler for learning rate adjustment. It prints training loss every 500 steps to monitor progress.

    Arguments:
        model (:obj:`torch.nn.Module`):
            The model being trained.
        training_loader (:obj:`torch.utils.data.DataLoader`):
            The DataLoader containing training data.
        epoch (:obj:`int`):
            The current epoch number.
        optimizer (:obj:`torch.optim.Optimizer`):
            The optimizer used for adjusting the model's weights.
        scheduler (:obj:`torch.optim.lr_scheduler`):
            The learning rate scheduler.
    """
    model.train()
    for _, data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_fn(outputs, targets)
        if _%500==0:
            print(f'Epoch: {epoch+1}, Training loss:  {loss.item()}')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()

def validation(model, dataloader, device):
    r"""Validates the model's performance on a separate dataset (valid/test).

    This function evaluates the model's performance by computing predictions for the data provided by the dataloader and comparing these predictions to
    the true labels.

    Arguments:
        model (:obj:`torch.nn.Module`):
            The model to evaluate.
        dataloader (:obj:`torch.utils.data.DataLoader`):
            The DataLoader containing validation or test data.
        device (:obj:`torch.device`):
            The device on which the model is running (e.g., cpu, cuda).

    Returns:
        :obj:`tuple`: A tuple containing two lists of predictions and actual labels.
    """
    model.eval()
    fin_targets = []
    fin_outputs = []

    with torch.no_grad():
        for _, data in tqdm(enumerate(dataloader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)

            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

def print_confusion_matrices(outputs, targets, target_names):
    r"""Generates and displays confusion matrices for multilabel classification.

    For each label, this function computes a confusion matrix using true and predicted labels and then visualizes it using a heatmap.

    Arguments:
        outputs (:obj:`list` of :obj:`int`):
            The predicted labels for each data point.
        targets (:obj:`list` of :obj:`int`):
            The true labels for each data point.
        target_names (:obj:`list` of :obj:`str`):
            The names of the labels.
    """
    for i, label in enumerate(target_names):
        cm = confusion_matrix(targets[:, i], outputs[:, i])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(label)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()