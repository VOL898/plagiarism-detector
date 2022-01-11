# Importing relevant libraries
import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

path = 'files' # Specifying the path for files
files = [filename for filename in os.listdir(path) if filename.endswith('.pdf')] # Saving all files that has the filetype pdf

contents = [] # Initilizing variable to store the contents of each file

# Reading all sample files to compare on
for file in files:

    file = open(path + '/' + file, 'rb')    # Opening the pdf file
    pdf_reader = PyPDF2.PdfFileReader(file) # Initializing reader to read the file
    page_count = pdf_reader.numPages        # Get the page count of the pdf file to loop on

    text = ''                               # Initilize empty variable to store the text per file

    # Looping for each page of the pdf file
    for page in range(page_count):
        page_obj = pdf_reader.getPage(page) # Get the page from the file
        text += page_obj.extractText()      # Extracting the text from the page
    
    contents.append(text)                   # Append the text extracted from the file to the variable that stores texts per file

vectorizer = TfidfVectorizer() # Preparing the vectorizer
vectors = vectorizer.fit_transform(contents).toarray() # Vectorizing the sample contents

index = files.index('checktext.pdf') # Getting the checktext.pdf file index from the list

# Obtaining the target file and vector from files and vectors
target_file = files[index]
target_vector = vectors[index]

s_vectors = list(zip(files, vectors)) # Pairing the files and vectors then putting it into a list

del s_vectors[index] # Deleting the target file and vector from the files and vectors list

results = [] # Initializing empty list to store the results

# For loop to get each sample file and its vector
for sample_file, sample_vector in s_vectors:

    similarity = cosine_similarity([target_vector, sample_vector])[0][1] * 100 # Getting the cosine similarity of the target vector to the sample vector multiplied by a hundred
    
    results.append((sample_file, similarity)) # Appending the target filename, the sample filename and the cosine similarity

sorted_results = sorted(results, key=lambda x: x[1], reverse=True) # Sort the results in descending order of its similarity score

# Displaying the results
print(f"{target_file} to:")
for data in sorted_results:
    print(f"{data[0]} has a similarity score of {data[1]:.2f} %")