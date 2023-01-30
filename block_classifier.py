from sre_parse import Tokenizer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import functional as FM
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchsampler import ImbalancedDatasetSampler
import os

PRETRAINED_BERT     = 'bert-base-uncased'
PRETRAINED_SCIBERT  = 'allenai/scibert_scivocab_uncased'
PRETRAINED_ELECTRA  = 'google/electra-base-discriminator'
PRETRAINED_T5       = 't5-small'

#FILE_DIR = '/workspace/paperassistant/backend/block_classifier'
FILE_DIR = os.path.dirname(__file__)
DATA_DIR = FILE_DIR + '/data'
MODEL_DIR = FILE_DIR + '/model'
RELEASE_DIR = FILE_DIR + '/release'
TASKS = ['auto-structurize', 'predict_next_structure']
DATA_MODELS = ['ignore', 'replace', 'ignore-56', 'replace-56', 'ignore-5']
MODELS = ['bert', 'scibert', 'electra', 't5']

model_path = ''
release_path = ''

class BlockDataset(Dataset):
    def __init__(self, data, model, tokenizer, label_vocab, max_seq_len):
        self.data_text  = [x[0] for x in data] # sentence
        self.data_label = [x[1] for x in data] # tag
        self.inputs = tokenizer(self.data_text, padding='max_length', truncation=True, 
                                return_tensors='pt', max_length = max_seq_len)
        self.outputs = [ label_vocab.get(x) for x in self.data_label ]
        self.model = model
    
    def __len__(self):
        return len(self.outputs)


    def __getitem__(self, idx):
        item = list()
        item.append(self.inputs.input_ids[idx])
        if self.model != 't5':
            item.append(self.inputs.token_type_ids[idx])
        item.append(self.inputs.attention_mask[idx])
        item.append(torch.tensor(self.outputs[idx]))
        #item = [
        #            # input size all same 
        #            self.inputs.input_ids[idx],
        #            self.inputs.token_type_ids[idx],
        #            self.inputs.attention_mask[idx],

        #            # output
        #            torch.tensor(self.outputs[idx])
        #]
        return item
    
    def get_labels(self):
        return self.data_label
    
class BlockDataModule(pl.LightningDataModule):
    def __init__(self, 
                 model: str = 'scibert', 
                 data_model: str = 'ignore', 
                 task: str = 'auto-structurize',
                 with_sampler: bool = True, 
                 batch_size: int = 8, 
                 max_seq_len: int = 128):
        super().__init__()
        self.model = model
        self.data_model = data_model
        self.task = task
        self.with_sampler = with_sampler
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    def prepare_data(self):
        # get the file name of each dataset
        fn_train, fn_valid, fn_test = self._get_fns()

        # dataset load to current CPU 
        self.train_data = self._load_data(f'{fn_train}')
        self.valid_data = self._load_data(f'{fn_valid}')
        self.test_data  = self._load_data(f'{fn_test}')

        # prepare tokenizer
        if self.model == 'bert':
            from transformers import BertTokenizerFast
            self.tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_BERT)

        elif self.model == 'scibert':
            from transformers import BertTokenizerFast
            self.tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_SCIBERT)
        
        elif self.model == 'electra':
            from transformers import ElectraTokenizer
            self.tokenizer = ElectraTokenizer.from_pretrained(PRETRAINED_ELECTRA)

        elif self.model == 't5':
            from transformers import T5Tokenizer 
            self.tokenizer = T5Tokenizer.from_pretrained(PRETRAINED_T5)

        # prepare label dictionary
        labels = sorted (list(set([x[1] for x in self.train_data] ) ) )
        self.label_vocab = { key: idx for idx, key in enumerate(labels) }
        print(self.label_vocab)
        self.num_class = len(self.label_vocab)

    def _get_fns(self):
        model = ''
        if self.data_model == 'base':
            pass
        elif self.data_model in DATA_MODELS:
            model = '_' + self.data_model
        else:
            print('invalid data_model:', self.data_model)
            raise ValueError
        
        task = ''
        if self.task == 'auto-structurize':
            pass
        elif self.task == 'predict_next_structure':
            task = '_reformed'
        else:
            print('invalid task:', self.task)
            raise ValueError

        fn_train = f'{DATA_DIR}/train{model}{task}.txt'
        fn_valid = f'{DATA_DIR}/valid{model}{task}.txt'
        fn_test = f'{DATA_DIR}/test{model}{task}.txt'

        return (fn_train, fn_valid, fn_test)

    def _load_data(self, fn):
        data = []
        with open(fn, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.rstrip() # remove '\n'
            text, label = line.split('\t')
            data.append( (text, label))
            
        return data
        
    def setup(self, stage = None):
        if stage in [None, 'train']:
            self.train_dataset = BlockDataset(self.train_data, self.model, self.tokenizer, self.label_vocab, self.max_seq_len)
            self.valid_dataset = BlockDataset(self.valid_data, self.model, self.tokenizer, self.label_vocab, self.max_seq_len)
        if stage in [None, 'test']:
            self.test_dataset  = BlockDataset(self.test_data, self.model, self.tokenizer, self.label_vocab, self.max_seq_len)

    def train_dataloader(self):
        if self.with_sampler:
            sampler = ImbalancedDatasetSampler(self.train_dataset)
            return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler, pin_memory=True)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
      
class BlockClassifier(pl.LightningModule):
    def __init__(self,
                 model,
                 num_class,
                 with_sampler,
                 release_path,
                 ### BERT #####
                 # --- optimizer specific -- #
                 #warmup_steps:  int = 500,
                 weight_decay:  float = 1e-2,
                 adam_epsilon:  float = 1e-8,
                 # --- trainer  specific -- #
                 learning_rate: float = 2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.release_path=release_path

        ## [text encoder]
        ## prepare pretrained - TRANSFORMER Model
        from transformers import BertModel
        if model == 'bert':
            self.encoder = BertModel.from_pretrained(PRETRAINED_BERT)
        elif model == 'scibert':
            self.encoder = BertModel.from_pretrained(PRETRAINED_SCIBERT)
        elif model == 'electra':
            self.encoder = BertModel.from_pretrained(PRETRAINED_ELECTRA)
        elif model == 't5':
            self.encoder = BertModel.from_pretrained(PRETRAINED_T5)
        else:
            print('model error:', model)
            raise ValueError

        # [to output]
        self.to_output = nn.Linear(self.encoder.config.hidden_size, num_class)
        nn.init.xavier_uniform_(self.to_output.weight)

        # loss
        self.criterton = nn.CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids, attention_mask):
        # attention_mask = attention_mask[:, None, None, :] # [B, 1, 1, seq_len] 
        if self.hparams.model == 't5':
            outputs = self.encoder(input_ids      = input_ids,
                                   attention_mask = attention_mask)
        else:
            outputs = self.encoder(input_ids      = input_ids,
                                   token_type_ids = token_type_ids,  
                                   attention_mask = attention_mask)

        hidden_states = outputs[0] # (batchsize, seq_len, dim) # outputs.hidden_states
        pooled_output = outputs[1] # outputs.pooler_output
        if self.hparams.model in ('electra', 't5'):
            logits = self.to_output(outputs.last_hidden_state[:, 0, :])
        else:
            logits = self.to_output(pooled_output)  # (batchsizes, dim)

        return logits
    
    def training_step(self, batch, batch_idx):
        if self.hparams.model == 't5':
            input_ids, attention_mask, label = batch
            logits = self(input_ids, None, attention_mask)
        else:
            input_ids, token_type_ids, attention_mask, label = batch
            logits = self(input_ids, token_type_ids, attention_mask)
        loss = self.criterton(logits, label.long())

        pred = F.softmax(logits, dim=-1)
        acc  = FM.accuracy(pred, label)
        metrics = {'train_acc': acc, 'train_loss': loss}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.hparams.model == 't5':
            input_ids, attention_mask, label = batch
            logits = self(input_ids, None, attention_mask)
        else:
            input_ids, token_type_ids, attention_mask, label = batch
            logits = self(input_ids, token_type_ids, attention_mask)
        loss = self.criterton(logits, label)

        prob = F.softmax(logits, dim=-1)
        acc  = FM.accuracy(prob, label)
        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics, prog_bar=True, on_epoch=True)
        return metrics

    def test_step(self, batch, batch_idx):
        if self.hparams.model == 't5':
            input_ids, attention_mask, label = batch
            logits = self(input_ids, None, attention_mask)
        else:
            input_ids, token_type_ids, attention_mask, label = batch
            logits = self(input_ids, token_type_ids, attention_mask)
        loss = self.criterton(logits, label.long())

        prob      = F.softmax(logits, dim=-1)
        #acc       = FM.accuracy(prob, label)
        #acc_top_3 = FM.accuracy(prob, label, top_k=3) 
        #f1        = FM.f1_score(prob, label)
        #metrics   = {'test_acc': acc, 
        #             'test_acc_top_3': acc_top_3,
        #             'test_f1': f1, 
        #             'test_loss': loss}
        #self.log_dict(metrics, prog_bar=True, on_epoch=True)

        #result = dict()
        #result['pred'] = prob.argmax(-1)
        #result['loss'] = loss
        #return result
        metrics = dict()
        metrics['prob'] = prob
        metrics['label'] = label
        return metrics

    def test_epoch_end(self, outputs):
        pred = result_collapse(outputs, 'prob')
        true = result_collapse(outputs, 'label')
        n_label = len(sorted(list(set(true.tolist()))))
        prec   = FM.precision(pred, true, average='weighted', num_classes=n_label)
        recall = FM.recall(pred, true, average='weighted', num_classes=n_label)
        f1     = FM.f1_score(pred, true, average='weighted', num_classes=n_label)
        acc    = FM.accuracy(pred, true)
        prec_top3   = FM.precision(pred, true, average='weighted', num_classes=n_label, top_k=3)
        recall_top3 = FM.recall(pred, true, average='weighted', num_classes=n_label, top_k=3)
        f1_top3     = FM.f1_score(pred, true, average='weighted', num_classes=n_label, top_k=3)
        acc_top3    = FM.accuracy(pred, true, top_k=3)
        metrics = {'test_recall': recall,
                   'test_f1': f1, 
                   'test_acc': acc,
                   'test_recall_top3': recall_top3,
                   'test_f1_top3': f1_top3, 
                   'test_acc_top3': acc_top3}
        self.log_dict(metrics)

        # in its version of lightning_logs
        from glob import glob
        import pandas as pd
        global model_path
        df_dict = {
            'recall': [recall.item()],
            'f1': [f1.item()],
            'acc': [acc.item()],
            'recall_top3': [recall_top3.item()],
            'f1_top3': [f1_top3.item()],
            'acc_top3': [acc_top3.item()]
        }
        version_list = sorted(glob(os.path.join(model_path, '*/version_*')))
        fn_metrics = version_list[-1] + '/metrics.csv'
        df = pd.DataFrame(df_dict)
        df.to_csv(fn_metrics, sep='\t')

        # dumping result
        pred = pred.argmax(-1)
        np.save(f'{self.release_path}/prediction.npy', pred.cpu().numpy())
        np.save(f'{self.release_path}/reference.npy', true.cpu().numpy())

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        #optimizer = torch.optim.Adam(self.parameters(), 
        #                            lr=self.hparams.learning_rate)
        return optimizer
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("N21")
        parser.add_argument('--learning_rate', type=float, default=2e-5)
        return parent_parser

from glob import glob
def process(mode, trainer, model, dm):
    if mode in ('train', None):
        trainer.fit(model, datamodule=dm)

    if mode in ('train', None):
        model_file_list = glob(os.path.join(model_path, '*.ckpt'))
        if len(model_file_list) != 0:
            best_fn_path = model_file_list[-1]
            best_fn_base = best_fn_path.split('/')[-1]
            version_list = sorted(glob(os.path.join(model_path, '*/version_*')))
            best_fn_to_move = version_list[-1] + '/' + best_fn_base
            import shutil
            shutil.move(best_fn_path, best_fn_to_move)

    if mode in ('test', None):
        model_file_list = glob(os.path.join(model_path, '*/*/*.ckpt'))
        #model_file_list = glob(os.path.join(model_path, '/workspace/paperassistant/backend/block_classifier/model/auto-structurize/ignore-5/scibert/16/lightning_logs/version_3/epoch=68-val_loss=0.48-val_acc=0.84.ckpt'))
        if len(model_file_list) != 0:
            best_fn_path = model_file_list[-1]
            model = BlockClassifier.load_from_checkpoint(best_fn_path) 
            result = trainer.test(model, dm.test_dataloader())
            print(result)
            dump_prediction_result(dm, release_path=release_path)


from argparse import ArgumentParser, BooleanOptionalAction
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from utils import dump_prediction_result, result_collapse
import os

SEED_VAL = 42

def cli_main():
    #pl.seed_everything(SEED_VAL, workers=True)
    pl.seed_everything(SEED_VAL)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--mode", help="mode - train, test, None(=train and test)", default=None, type=str)
    parser.add_argument("--model", help="model - bert, scibert", default="scibert", type=str)
    parser.add_argument("--data_model", help="data model - base, ignore, replace, ignore-56, replace-56, ignore-5", default="ignore-5", type=str)
    parser.add_argument("--task", help="task - auto-structurize, predict_next_structure", default="auto-structurize", type=str)
    parser.add_argument("--with_sampler", action=BooleanOptionalAction, default=False, type=bool, help="put this if use sampler")
    parser.add_argument('--max_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_seq_len', default=128, type=int)
    parser.add_argument("--gpu_id", help="ID(s) of the gpu(s) to use", default='1', type=str)
    parser.add_argument("--gpu_method", help="How to use multi-gpus", default=None, type=str)

    parser = pl.Trainer.add_argparse_args(parser)
    parser = BlockClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    global model_path
    model_path = f'{MODEL_DIR}/{args.task}/{args.data_model}/{args.model}/{args.batch_size}'
    try:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
    except OSError:
        print(f'ERROR: Creating directory {model_path}')

    global release_path 
    release_path = f'{RELEASE_DIR}/{args.model}/{args.batch_size}'
    try:
        if not os.path.exists(release_path):
            os.makedirs(release_path)
    except OSError:
        print(f'ERROR: Creating directory {release_path}')

    # -------------
    # data
    # -------------
    dm = BlockDataModule.from_argparse_args(args)
    dm.prepare_data()
    dm.setup(stage=args.mode)

    # --------------
    # model
    # --------------
    earlystopping = EarlyStopping(monitor='val_loss', patience=3)
    checkpoint = ModelCheckpoint(monitor='val_loss', dirpath=model_path,
                                 filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}'
                                 )
    logger = TensorBoardLogger(save_dir=model_path)
    model = BlockClassifier(model=args.model,
                            num_class=dm.num_class,
                            with_sampler=args.with_sampler,
                            release_path=release_path,
                            learning_rate=args.learning_rate
                            )
    trainer = pl.Trainer(max_epochs=args.max_epoch, 
                         callbacks=[earlystopping, checkpoint],
                         logger=logger,
                         gpus=args.gpu_id,  # if you have gpu -- set number, otherwise zero   
                         strategy=args.gpu_method,
                         replace_sampler_ddp=(not args.with_sampler)
                         )
    
    process(args.mode, trainer, model, dm)

if __name__ == '__main__':
    cli_main()
