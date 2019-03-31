import numpy as np

def translate_sentences(data, encoder_model, decoder_model, max_len, idx_to_word_dict, word_to_idx_dict):
    
    sentences = []
    init = "<S>"
    
    for sentence in data:
        
        cnt = 0
        words = []
        sentence_ = [word_to_idx_dict[x] for x in sentence.split(" ") if x in word_to_idx_dict]
        state = encoder_model.predict(sentence_)
        
        while init != "</S>" and cnt <= max_len + 1:
            indeces, state = decoder_model.predict(word_to_idx_dict[init], state)
            index = np.argmax(indeces) # please check indeces.shape at first
            init = idx_to_word_dict[index]
            words.append(init)
            cnt += 1
        sentences.append(words[:-1])
    
    return sentences