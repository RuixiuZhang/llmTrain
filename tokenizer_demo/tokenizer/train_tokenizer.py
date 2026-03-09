from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import Sequence, NFKC, Lowercase
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# initialize tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# normalization
tokenizer.normalizer = Sequence([
    NFKC(),
    Lowercase()
])

# Byte-level pre-tokenizer
tokenizer.pre_tokenizer = ByteLevel()

# trainer
trainer = BpeTrainer(
    vocab_size=16000,
    min_frequency=2,
    special_tokens=[
        "[PAD]",
        "[UNK]",
        "[BOS]",
        "[EOS]"
    ]
)

# train
files = ["corpus.txt"]
tokenizer.train(files, trainer)

# decoder
tokenizer.decoder = ByteLevelDecoder()

# save
tokenizer.save("tokenizer.json")

print("Tokenizer training finished.")
print("Vocab size:", tokenizer.get_vocab_size())