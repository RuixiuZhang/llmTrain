from collections import Counter

class BPETokenizer:
    def __init__(self,vocab_size=1000):
        self.vocab_size=vocab_size
        self.merges=[]
        self.vocab={}

    def build_vocab(self,corpus):
        vocab=Counter()
        for line in corpus:
            for word in line.strip().split():
                tokens=tuple(list(word)+["</w>"])
                vocab[tokens]+=1
        return vocab

    def get_pair_stats(self,vocab):
        pairs=Counter()
        for word,freq in vocab.items():
            for i in range(len(word)-1):
                pairs[(word[i],word[i+1])]+=freq
        return pairs

    def merge_pair(self,pair,vocab):
        new_vocab={}
        for word,freq in vocab.items():
            new_word=[]
            i=0
            while i<len(word):
                if i<len(word)-1 and word[i]==pair[0] and word[i+1]==pair[1]:
                    new_word.append(word[i]+word[i+1])
                    i+=2
                else:
                    new_word.append(word[i])
                    i+=1
            new_vocab[tuple(new_word)]=freq
        return new_vocab

    def train(self,corpus):
        vocab=self.build_vocab(corpus)
        for i in range(self.vocab_size):
            pairs=self.get_pair_stats(vocab)
            if not pairs: break
            best=pairs.most_common(1)[0][0]
            vocab=self.merge_pair(best,vocab)
            self.merges.append(best)
            if i%100==0:
                print("Merge",i,best)
        self.vocab=vocab

    def encode_word(self,word):
        tokens=list(word)+["</w>"]
        while True:
            pairs=[(tokens[i],tokens[i+1]) for i in range(len(tokens)-1)]
            mergeable=[p for p in pairs if p in self.merges]
            if not mergeable: break
            pair=mergeable[0]
            i=pairs.index(pair)
            tokens=tokens[:i]+[pair[0]+pair[1]]+tokens[i+2:]
        return [t.replace("</w>","") for t in tokens if t!="</w>"]

    def encode(self,text):
        words=text.split()
        result=[]
        for w in words:
            result.extend(self.encode_word(w))
        return result

    def decode(self,tokens):
        return "".join(tokens)

    def save(self,path):
        with open(path,"w") as f:
            for a,b in self.merges:
                f.write(a+" "+b+"\n")


import random

base_sentences=[
"machine learning is powerful",
"deep learning models are useful",
"artificial intelligence is the future",
"natural language processing is interesting",
"large language models generate text",
"neural networks learn patterns",
"data science uses statistics",
"python is popular for machine learning",
"optimization improves model performance",
"training data is important",

"machine learning algorithms analyze data",
"deep neural networks recognize images",
"artificial intelligence changes industries",
"natural language models understand text",
"large datasets improve model training",
"neural networks require large datasets",
"data scientists analyze complex datasets",
"python libraries support machine learning",
"optimization algorithms improve efficiency",
"training neural networks takes time",

"machine learning models predict outcomes",
"deep learning methods solve complex tasks",
"artificial intelligence powers many systems",
"natural language processing analyzes text",
"large language models generate answers",
"neural networks model complex relationships",
"data science combines math and computing",
"python tools simplify data analysis",
"optimization techniques improve training",
"training models requires computing power",

"machine learning improves recommendation systems",
"deep learning powers computer vision",
"artificial intelligence supports automation",
"natural language systems translate languages",
"large language models summarize documents",
"neural networks classify images",
"data science drives business insights",
"python frameworks accelerate development",
"optimization methods tune model parameters",
"training datasets must be clean",

"machine learning techniques analyze patterns",
"deep neural architectures improve performance",
"artificial intelligence supports decision making",
"natural language models answer questions",
"large language models assist programming",
"neural networks detect anomalies",
"data scientists explore large datasets",
"python is widely used in science",
"optimization algorithms find best solutions",
"training models requires experimentation",

"machine learning enables predictive analytics",
"deep learning models analyze images",
"artificial intelligence powers search engines",
"natural language processing extracts information",
"large language models generate code",
"neural networks perform feature extraction",
"data science improves decision systems",
"python ecosystems support AI development",
"optimization improves convergence speed",
"training models requires many iterations",

"machine learning pipelines process data",
"deep neural networks learn hierarchical features",
"artificial intelligence improves robotics",
"natural language models assist chat systems",
"large language models write summaries",
"neural networks require backpropagation",
"data scientists build predictive models",
"python notebooks support experiments",
"optimization helps avoid local minima",
"training neural networks requires patience",

"machine learning improves fraud detection",
"deep learning models classify speech",
"artificial intelligence assists healthcare",
"natural language systems detect sentiment",
"large language models generate explanations",
"neural networks approximate functions",
"data science supports analytics pipelines",
"python packages simplify experiments",
"optimization strategies improve accuracy",
"training models benefits from good data",

"machine learning supports recommendation engines",
"deep neural networks analyze signals",
"artificial intelligence assists navigation",
"natural language processing supports chatbots",
"large language models assist writing",
"neural networks support time series prediction",
"data science integrates statistics and computing",
"python code powers many AI systems",
"optimization algorithms reduce loss",
"training models requires evaluation",

"machine learning discovers hidden patterns",
"deep learning models detect objects",
"artificial intelligence enables smart assistants",
"natural language models interpret meaning",
"large language models assist research",
"neural networks support complex modeling",
"data science powers modern analytics",
"python tools enable rapid prototyping",
"optimization methods improve convergence",
"training models improves with experience"
]

corpus=[]
for _ in range(50000):
    corpus.append(random.choice(base_sentences))

tokenizer=BPETokenizer(vocab_size=100)
tokenizer.train(corpus)

print("\nTest tokenization:")
print(tokenizer.encode("machine learning"))
print(tokenizer.encode("artificial intelligence"))
print(tokenizer.encode("neural networks"))