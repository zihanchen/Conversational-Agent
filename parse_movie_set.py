'''
Parse the movie subtitles data set into a format that satisfies the following:
- All lower case
- Every sentence has 10 words
  (pad with the word "pad" if len < 10 and truncate if len > 10)
'''

import string

class MovieSet:
    def parse_movie_set(self, datatype='train'):
        file_open = open('data/cornell movie-dialogs corpus/movie_lines.txt', 'r');

        '''
        # Parse first 2000 lines for testing
        for i in range(2000):
            line = file_open.readline()
            strings = line.split(' +++$+++ ')
            file_write.write(strings[-1])
        '''

        # Parse Q/A pairs with exactly 10 words (truncate length)
        # pad if < 10 and truncate if > 10
        trunc_len = 10
        if datatype == 'train':
            file_write = open('parsed_train.txt', 'w');
            for i in range(2400):
                line = file_open.readline().split(' +++$+++ ')[-1].lstrip(' ').rstrip('\n')
                punc = string.punctuation.replace("\'","")
                newline = ""
                for i in range(len(line)):
                    if not line[i].isalpha() and line[i] != "'":
                        newline += " "
                    else:
                        newline += line[i]
                line = ' '.join(newline.split())
                line_len = len(line.split())
                if line_len > trunc_len:
                    line = ' '.join(line.split(' ')[0:trunc_len])
                else:
                    line = ' '.join(line.split(' ') + [' PAD'] * (trunc_len - line_len))
                file_write.write(' '.join(line.lower().split()) + '\n')
            file_write.close()
        elif datatype == 'all':
            file_write = open('parsed_all.txt', 'w');
            for i in range(3000):
                line = file_open.readline().split(' +++$+++ ')[-1].lstrip(' ').rstrip('\n')
                punc = string.punctuation.replace("\'","")
                newline = ""
                for i in range(len(line)):
                    if not line[i].isalpha() and line[i] != "'":
                        newline += " "
                    else:
                        newline += line[i]
                line = ' '.join(newline.split())
                line_len = len(line.split())
                if line_len > trunc_len:
                    line = ' '.join(line.split(' ')[0:trunc_len])
                else:
                    line = ' '.join(line.split(' ') + [' PAD'] * (trunc_len - line_len))
                file_write.write(' '.join(line.lower().split()) + '\n')
            file_write.close()
        else:
            file_write = open('parsed_test_questions.txt', 'w');
            file_answers = open('parsed_test_answers.txt', 'w');
            for i in range(3000):

                line = file_open.readline().split(' +++$+++ ')[-1].lstrip(' ').rstrip('\n')
                if (i < 2400): continue
                punc = string.punctuation.replace("\'","")
                newline = ""
                for j in range(len(line)):
                    if not line[j].isalpha() and line[j] != "'":
                        newline += " "
                    else:
                        newline += line[j]
                line = ' '.join(newline.split())
                line_len = len(line.split())
                if line_len > trunc_len:
                    line = ' '.join(line.split(' ')[0:trunc_len])
                else:
                    line = ' '.join(line.split(' ') + [' PAD'] * (trunc_len - line_len))
                if (i % 2 == 0):
                    file_answers.write(' '.join(line.lower().split()) + '\n')
                else:
                    file_write.write(' '.join(line.lower().split()) + '\n')
            file_write.close()
            file_answers.close()
        file_open.close()



m = MovieSet()
m.parse_movie_set('test')
