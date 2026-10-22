"""Version-independent T5 SentencePiece encoding, verified against the actual model."""
import sentencepiece as spm
import torch

class FrozenTokenizer:
    def __init__(self,path):
        self.sp=spm.SentencePieceProcessor(model_file=str(path))
        if self.sp.vocab_size()!=32000 or self.sp.unk_id()!=2 or self.sp.pad_id()!=0 or self.sp.eos_id()!=1:
            raise ValueError('Unexpected T5 SentencePiece contract')
    def encode(self,text):return self.sp.encode(text,out_type=int)+[self.sp.eos_id()]
    def __call__(self,texts,**ignored):
        tokens=[self.encode(t) for t in texts];ids=torch.full((len(tokens),max(map(len,tokens))),0,dtype=torch.long);mask=torch.zeros_like(ids)
        for i,t in enumerate(tokens):ids[i,:len(t)]=torch.tensor(t);mask[i,:len(t)]=1
        return dict(input_ids=ids,attention_mask=mask)

def tokenizer_checks(path):
    t=FrozenTokenizer(path)
    sentences=['A person walks forward.','A person sits down.','A person raises their left hand.']
    expected=[[71,568,10681,1039,5,1],[71,568,2561,7,323,5,1],[71,568,3033,7,70,646,609,5,1]]
    actual=[t.encode(s) for s in sentences]
    if actual!=expected:raise ValueError('T5 tokenizer known-answer mismatch')
    return dict(vocabulary=32000,token_ids=actual,distinct_same_word_count_sentences=actual[0]!=actual[1],known_answers_pass=True,
                implementation='original frozen spiece.model directly through SentencePieceProcessor + EOS; pad=0, unk=2; no Transformers tokenizer API assumptions')
