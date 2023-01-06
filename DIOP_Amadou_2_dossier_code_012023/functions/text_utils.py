import time
from sklearn import cluster, metrics
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np



# Tokenizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Stop words

stop_w = list(set(stopwords.words('english'))) + ['[', ']', ',', '.', ':', '?', '(', ')']

def tokenizer_fct(sentence) :
    # print(sentence)
    sentence_clean = sentence.replace('-', ' ').replace('+', ' ').replace('/', ' ').replace('#', ' ')
    word_tokens = word_tokenize(sentence_clean)
    return word_tokens



def stop_word_filter_fct(list_words) :
    filtered_w = [w for w in list_words if not w in stop_w]
    filtered_w2 = [w for w in filtered_w if len(w) > 2]
    return filtered_w2

# lower case et alpha
def lower_start_fct(list_words) :
    lw = [w.lower() for w in list_words if (not w.startswith("@")) 
    #                                   and (not w.startswith("#"))
                                       and (not w.startswith("http"))]
    return lw

# Lemmatizer (base d'un mot)


def lemma_fct(list_words) :
    lemmatizer = WordNetLemmatizer()
    lem_w = [lemmatizer.lemmatize(w) for w in list_words]
    return lem_w

# Fonction de préparation du texte pour le bag of words (Countvectorizer et Tf_idf, Word2Vec)
def transform_bow_fct(desc_text) :
    word_tokens = tokenizer_fct(desc_text)
    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(sw)
    # lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(lw)
    return transf_desc_text

# Fonction de préparation du texte pour le bag of words avec lemmatization
def transform_bow_lem_fct(desc_text) :
    word_tokens = tokenizer_fct(desc_text)
    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(sw)
    lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(lem_w)
    return transf_desc_text

# Fonction de préparation du texte pour le Deep learning (USE et BERT)
def transform_dl_fct(desc_text) :
    word_tokens = tokenizer_fct(desc_text)
#    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(word_tokens)
    # lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(lw)
    return transf_desc_text

#data_T['sentence_bow'] = data_T0['text'].apply(lambda x : transform_bow_fct(x))
#data_T['sentence_bow_lem'] = data_T0['text'].apply(lambda x : transform_bow_lem_fct(x))
#data_T['sentence_dl'] = data_T0['text'].apply(lambda x : transform_dl_fct(x))
#data_T.shape
 
# Calcul Tsne, détermination des clusters et calcul ARI entre vrais catégorie et n° de clusters
def ARI_fct(features,all_labels,labels_encoded,perplexity,n_components,random_state) :
    time1 = time.time()
    num_labels=len(all_labels)
    
    tsne = manifold.TSNE(
        #n_components=n_components,
        init='pca',
        random_state=random_state,
        learning_rate='auto',
        perplexity=perplexity
    )
    X_tsne = tsne.fit_transform(features)
    
    # Détermination des clusters à partir des données après Tsne 
    cls = cluster.KMeans(n_clusters=num_labels, n_init=100, random_state=random_state)
    cls.fit(X_tsne)
    
  
    ARI = np.round(metrics.adjusted_rand_score(labels_encoded, cls.labels_),4)
    time2 = np.round(time.time() - time1,0)
    print("ARI : ", ARI, "time : ", time2)
    
    return ARI, X_tsne, cls.labels_


# visualisation du Tsne selon les vraies catégories et selon les clusters
def TSNE_visu_fct(X_tsne,all_labels, labels_encoded, labels, ARI,title1,title2,title3='Cluster Tsne',labels_inverse=0) :
    #print('labels',labels)
    labels_legend = labels
    if labels_inverse!=0:
       labels_legend=labels_inverse
    fig = plt.figure(figsize=(20,6))
    
    ax = fig.add_subplot(121)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c=all_labels, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=labels_encoded, loc="best", title="Categorie")
    plt.title(title1)
    #print('labels',labels)
    ax = fig.add_subplot(122)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c=labels, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=set(labels_legend), loc="best", title=title3)
    plt.title(title2)
    
    plt.show()

    print("ARI : ", ARI)


def compare_clusterization(features,n_clusters,real_labels,real_labels_encoded,labels_encoded,perplexity,random_state,title1,title2):


    time1 = time.time()
     
    
    tsne = manifold.TSNE(
        #n_components=n_components,
        init='pca',
        random_state=random_state,
        learning_rate='auto',
        perplexity=perplexity
    )
    X_tsne = tsne.fit_transform(features)
    
    # Détermination des clusters à partir des données après Tsne 
    cls = cluster.KMeans(n_clusters=n_clusters, n_init=100, random_state=random_state)
    cls.fit(X_tsne)
    
  
    ARI = np.round(metrics.adjusted_rand_score(labels_encoded, cls.labels_),4)
    time2 = np.round(time.time() - time1,0)
    
    fig = plt.figure(figsize=(20,6))
    ax = fig.add_subplot(121)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c=real_labels_encoded, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=real_labels, loc="best", title="Categorie")
    plt.title(title1)
    #print('labels',labels)
    ax = fig.add_subplot(122)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c=cls.labels_, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=set(cls.labels_), loc="best", title=title2)
    plt.title(title2)
    
    plt.show()
    print("ARI : ", ARI, "time : ", time2)
