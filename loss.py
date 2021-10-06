from sentence_transformers import SentenceTransformer, util

def SBERT_loss(true_description, receiver_output):

    """
    Estimate the Cosine similarity among sentences
    using SBERT
    https://www.sbert.net/docs/usage/semantic_textual_similarity.html
    https://arxiv.org/abs/1908.10084
    https://github.com/UKPLab/sentence-transformers
    """
            
    model = SentenceTransformer('all-MiniLM-L6-v2')

    emb1 = model.encode(receiver_output)
    emb2 = model.encode(true_description)

    loss = -util.cos_sim(emb1, emb2)
    print(loss)
    
    return loss