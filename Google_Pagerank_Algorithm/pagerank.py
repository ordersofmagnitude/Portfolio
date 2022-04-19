import os
import random
import re
import sys
import numpy as np

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
       sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl("corpus0")
    print(corpus)
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Returns transition model.
    """
    tm = {}
    
    if len(corpus[page]) == 0:
        for c in corpus:       
            tm[c] = 1/ len(corpus)
    
    else:
        for c in corpus:
            tm[c] = (1-damping_factor)/len(corpus)
            
        for webpage in corpus[page]:
            tm[webpage] += damping_factor / len(corpus[page])
    
    return tm


def sample_pagerank(corpus, damping_factor, n):
    """
    Returns pagerank values via sampling
    """
    rank = {}
    count = 0
    
    #initialize
    for c in corpus:
        rank[c] = 0.0
    
    while count < n:
        
        #randomly generate 1st page
        if count == 0:
            pg = np.random.choice(list(corpus.keys()))
            
            #generate transition model for page
            transition = transition_model(corpus, pg, DAMPING)
            
            #increment p & count
            rank[pg] += 1
            count += 1
            
        else:
            next_page = random.choices(list(transition.keys()), weights = list(transition.values()))            
            transition = transition_model(corpus, next_page[0], DAMPING)
            
            rank[next_page[0]] += 1
            count += 1
            
    for item in rank:
        rank[item] = rank[item]/ n
        
    return rank


def iterate_pagerank(corpus, damping_factor):
    """
    Returns pagerank values via iteration and updates values until convergence
    """
    
    ranked = {}
    previous = {}
    
    #initialize
    for link in corpus:
        ranked[link] = (1 - damping_factor)/ len(corpus)
    
    converged = False
    while not converged:
            
        for link in corpus:  
                     
            previous = ranked.copy()
            
            #update based on damping factor formula
            for k, v in corpus.items():
                if link in v and k != link:
                    ranked[link] += damping_factor * (ranked[k]/len(v))
            
        #convergence test
        for key in ranked:
            if np.abs(ranked[key] - previous[key]) != 0.001:
                converged = False
        
        converged = True    
    
    return ranked

if __name__ == "__main__":
    main()
