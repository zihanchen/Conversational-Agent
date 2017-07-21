import word2vec
import numpy as np
import parse_movie_set
import random

created = False

def create_model():
    global created
    if (created):
        model = word2vec.load('movie_all.bin')
        return model
    else:
        movie_set = parse_movie_set.MovieSet()
        movie_set.parse_movie_set('train')
        movie_set.parse_movie_set('test')
        movie_set.parse_movie_set('all')
        word2vec.word2phrase('parsed_all.txt', 'parsed_all_phrases.txt', verbose=True)
        word2vec.word2vec('parsed_all_phrases.txt', 'movie_all.bin', size=100, verbose=True)
        model = word2vec.load('movie_all.bin')
        created = True
        return model

def create_embedding(datatype='train'):
    model = create_model()
    if datatype == 'train':
        mat = []
        fo = open('parsed_train.txt', 'r')
        all_train = fo.readlines()

        for i in range(len(all_train)):
            if ( i == len(all_train) - 1): continue
            answer = all_train[i].rstrip('\n')
            question = all_train[i+1].rstrip('\n')
            line = question + ' ' + answer
            #print line
            print "%d / %d" % (i + 1, len(all_train) - 1)
            line = line.split(' ')
            for word in line:
                try:
                    c = model[word]
                except:
                    mat.append([0] * 100)
                    continue
                mat.append(model[word].tolist())
        mat = np.array(mat)

        # Reshape to conform with input placeholder
        # (Avoid Tensorflow ValueError)
        mat = np.reshape(mat, (-1, 2000))
        fo.close()
        np.savetxt('movietrain.txt', mat, fmt='%.4f', delimiter=',')
    else:
        mat = []
        fq = open('parsed_test_questions.txt', 'r')
        fa = open('parsed_test_answers.txt', 'r')
        f_a = open('parsed_all.txt', 'r')
        questions = fq.readlines()
        answers = fa.readlines()
        all_pairs = f_a.readlines()
        count = 0;
        for question in questions:
            for i in range(3):
                answer = ""
                if (i == 0):
                    answer = answers[count].rstrip('\n')
                    count += 1
                else:
                    rand_num = random.randint(0, 2999)
                    answer = all_pairs[rand_num].rstrip('\n')
                line = question.rstrip('\n') + ' ' + answer
                #print line
                line = line.split(' ')
                for word in line:
                    try:
                        c = model[word]
                    except:
                        mat.append([0] * 100)
                        continue
                    mat.append(model[word].tolist())
        mat = np.array(mat)
        mat = np.reshape(mat, (-1, 2000))
        fq.close()
        fa.close()
        f_a.close()
        np.savetxt('movietest.txt', mat, fmt='%.4f', delimiter=',')

if __name__ == '__main__':
    create_embedding('train')
    create_embedding('test')
