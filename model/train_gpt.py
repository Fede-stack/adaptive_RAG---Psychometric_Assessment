import gc 
from tqdm import tqdm

def train(cosine, type_embs):
    predictions_tot_gpt = []
    error_indices = [] 
    for j in tqdm(range(len(docss)), desc="Processing Users"):
        if type_embs == 1:
            doc_embeddings = get_embedding(docss[j], model) #with SentenceTransformers
        elif type_embs == 2:
            doc_embeddings = get_embedding(docss[j], model, tokenizer) #with transformers 
        else:
            doc_embeddings = np.array([get_embedding(doc) for doc in docss[j]])

        documents_retrieved = []
        for item in items_embs:
                    try:
                        embs = np.concatenate((np.array(item).reshape(1, -1), doc_embeddings))
                        #print(embs.shape)
                        data = Data(embs)
                        ids, kstars = return_ids_kstar_binomial(data, doc_embeddings, initial_id=None, Dthr=6.67, r='opt', n_iter=10)
                        nns = find_single_k_neighs(embs, 0, kstars[0], cosine)

                        documents_retrieved.append(np.array(docss[j])[np.array(nns) - 1].tolist())
                    except ValueError as e:
                        if "array must not contain infs or NaNs" in str(e):
                            error_indices.append((i, j))
                        else:
                            raise e

        zero_shot_scores_gpt = []
        for i in range(21):
            posts = np.unique(list(itertools.chain.from_iterable(documents_retrieved[(i*4):((i+1)*4)]))).tolist()
            posts.sort()
            posts_final = '\n '.join(posts)
            
            content = ''.join([str(i)+' ' + item + '\n ' for i, item in enumerate(bdi_items[i])])
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": prompt}
                ], temperature=0, max_tokens=3
            )
            zero_shot_scores_gpt.append(response.choices[0].message.content)

        predictions_gpt = np.array(zero_shot_scores_gpt)#.astype(np.int32)
        predictions_tot_gpt.append(predictions_gpt)
 
        gc.collect()

    
    return predictions_tot_gpt
