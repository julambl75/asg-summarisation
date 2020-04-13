import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from more_itertools import grouper


class Preprocessor:
    def __init__(self, device):

        # We can update this to use more complex embeddings
        print("Initialising preprocessor...")

        self.mask_prob = 0.2

        self.device = device

        self.bert_tokeniser = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)
        self.bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.bert_model.eval()
        self.bert_model = self.bert_model.to(device)

        print("Completed preprocessor initialisation")

    def bert_preprocess(self, ens, des):
        idxs_en = []
        for line in ens:
            tokens_ = self.bert_tokeniser.tokenize(line)
            idxs_en.append(self.bert_tokeniser.convert_tokens_to_ids(tokens_))

        idxs_de = []
        for line in des:
            tokens_ = self.bert_tokeniser.tokenize(line)
            idxs_de.append(self.bert_tokeniser.convert_tokens_to_ids(tokens_))

        return idxs_en, idxs_de

    def bert_preprocess_embs(self, ens, des):
        batch_size = 100

        processed_ens = []
        for batch_ens in grouper(batch_size, ens, None):
            ens_, att_en, pad_idxs = self._bert_preprocess_lines(batch_ens)
            hidden = self.bert_embds_batch(ens_, att_en, pad_idxs)
            processed_ens.extend(hidden)

        processed_des = []
        for batch_des in grouper(batch_size, des, None):
            des_, att_de, pad_idxs = self._bert_preprocess_lines(batch_des)
            hidden = self.bert_embds_batch(des_, att_de, pad_idxs)
            processed_des.extend(hidden)

        return processed_ens, processed_des

    def _bert_preprocess_lines(self, lines):
        idxs = []
        for line in lines:
            if line:
                tokens_ = self.bert_tokeniser.tokenize("[CLS]" + line + "[SEP]")
                idxs.append(self.bert_tokeniser.convert_tokens_to_ids(tokens_))

        new_idxs = []
        pad_idxs = []
        max_len = max([len(idxs_) for idxs_ in idxs])

        # Pad idxs with [PAD] token
        att_mask = torch.ones(len(lines), max_len, dtype=torch.int64)
        for idx, idxs_ in enumerate(idxs):
            len_idxs = len(idxs_)
            diff = max_len - len_idxs
            if diff > 0:
                idxs_.extend([0] * diff)
                att_mask[idx, len(idxs_):] = 0

            new_idxs.append(idxs_)
            pad_idxs.append(len_idxs)

        tokens_tensor = torch.tensor(new_idxs)
        return tokens_tensor, att_mask, pad_idxs

    def bert_embds_batch(self, tokens, att, pad_idxs):
        tokens = tokens.to(self.device)
        att = att.to(self.device)
        with torch.no_grad():
            hidden, _ = self.bert_model(tokens, att, output_all_encoded_layers=False)

        hidden = [embs[:pad_idxs[idx], :] for idx, embs in enumerate(list(hidden))]

        return hidden
