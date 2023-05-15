import torch
import torch.nn as nn

device = torch.device('cpu')

# training data
# using Cornell Movie-Dialogs Corpus
# https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
def training_data():
    # Download the data
    import urllib.request
    url = 'https://www.dropbox.com/s/2fdn26rj6h9bpvl/cornell_movie_dialogs_corpus.zip?dl=1'
    urllib.request.urlretrieve(url, filename='cornell_movie_dialogs_corpus.zip')

    # Extract the data
    import zipfile
    with zipfile.ZipFile('cornell_movie_dialogs_corpus.zip', 'r') as zip_ref:
        zip_ref.extractall()

    # Import the data
    import os
    import pandas as pd

    # Import the data
    lines = open('cornell movie-dialogs corpus/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
    conv_lines = open('cornell movie-dialogs corpus/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

    # Create a dictionary to map each line's id with its text
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]

    # Create a list of all of the conversations' lines' ids.
    convs = []
    for line in conv_lines[:-1]:
        _line = line.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
        convs.append(_line.split(','))

    # Sort the sentences into questions (inputs) and answers (targets)
    questions = []
    answers = []
    for conv in convs:
        for i in range(len(conv) - 1):
            questions.append(id2line[conv[i]])
            answers.append(id2line[conv[i + 1]])

    # Compare lengths of questions and answers
    print(len(questions))
    print(len(answers))

    # Define path to new file
    datafile = os.path.join("cornell movie-dialogs corpus", "formatted_movie_lines.txt")

    # Write new csv file
    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))
    # Write headers to the new file
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in zip(questions, answers):
            writer.writerow(pair)

    # Print a sample of lines
    datafile = os.path.join("cornell movie-dialogs corpus", "formatted_movie_lines.txt")
    with open(datafile, "rb") as file:
        lines = file.readlines()

    for line in lines[:8]:
        print(line)
