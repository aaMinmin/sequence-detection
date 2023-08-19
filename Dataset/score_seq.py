import csv
# open the two input files
for i in range(50,150):
    with open('./rand_dnaseq/rand_dna_seq_'+str(i)+'.csv') as f2, open('./pred/pred_score_seq'+str(i)+'.csv') as f1:
      # create csv reader objects
      cf1 = csv.reader(f1)
      cf2 = csv.reader(f2)
      # skip the headers
      next(cf1)
      next(cf2)
      # open the output file
      with open('./together/seq'+str(i)+'.csv', 'w',newline = '') as out:
        # create a csv writer object
        cw = csv.writer(out)
        # loop through the rows of both input files
        for row1, row2 in zip(cf1, cf2):
          # write the first column of each row to the output file
          cw.writerow([row1[0], row2[0]])