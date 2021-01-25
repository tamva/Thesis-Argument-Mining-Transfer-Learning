import pandas as pd
from time import sleep
import tqdm
import sys

# transformation
# lin = "TheNewLINE	O	O	N	SP: 2	Sentence: 2	Doc: essay004"
# fin = open('C:/Users/athanasis/Desktop/Semester_C/Dataset/dataset_conll/train_no_double_quotes.txt')
# # fout = open('C:/Users/athanasis/Desktop/Semester_C/Dataset/Binary-Conll/bn-train.txt')
# for line in fin:
#    # For Python3, use print(line)
#    print (line)
#
#    if line in ['\n', '\r\n']:
#     with open("C:/Users/athanasis/Desktop/Semester_C/Dataset/Binary-Conll/bb_train_no_double_quotes.txt", "a") as myfile:
#         deli = '-//-'
#         # myfile.write("{}\n".format(deli ))
#         myfile.write( lin)
#         myfile.write("\n")
#    else:
#     with open("C:/Users/athanasis/Desktop/Semester_C/Dataset/Binary-Conll/bb_train_no_double_quotes.txt", "a") as myfile:
#         # myfile.write("\n")
#         myfile.write(line)
#
# fin.close()
############

df = pd.read_csv('C:/Users/athanasis/Desktop/Semester_C/Dataset/Binary-Conll/bb_test_no_double_quotes.txt',sep = '\t',header=None)
print(df.loc[[0]][3])
print(df.loc[[0]][0])
range = range(len(df))
print(df.head(17))

sentences = []
arg_list = []
sentence  = ""
arg_value = 0

for index,line in df.iterrows():
        # sys.stdout.write('\r')
        # sys.stdout.write("[%-20s] %d%%" % ('=' * line, 5 * line))
        # sys.stdout.flush()
        sleep(0.05)
        print(sentences)
        print(arg_list)
        # word = df.loc[[index]][0]

        # print("\r{0}".format((float(line) / range) * 100))
        # if df.loc[[index]][0] == ('\n', '\r\n'):
        letters = df.loc[[index]][0]
        first_four_letters = letters[:4]
        # word = str(letters)
        word = letters.values[0]
        if word == 'TheNewLINE':
        # if not df.loc[[index]][0].strip() :
           sentences.append(sentence)
           arg_list.append(arg_value)
           sentence = ""
           arg_value = "nan"
        else:
           kapa = '"'
           if df.loc[[index]][0].values[0] == kapa:
               sentence = sentence + " " + "'"
           else:
               sentence = sentence + " " + df.loc[[index]][0].values[0]
               if df.loc[[index]][3].values[0] == 'Y':
                    arg_value = 1
               else:
                   arg_value = 0

binary_data = {'Sentence':sentences, 'Argument':arg_list }

binary_df = pd.DataFrame(binary_data, columns= ['Sentence','Argument'])
# bar.finish()
file_name = 'binary-dev'
print(binary_df.head())
binary_df.to_csv('C:/Users/athanasis/Desktop/Semester_C/Dataset/Binary-Conll/Final Data/binary-test_no_double_quotes.txt',sep='\t')

