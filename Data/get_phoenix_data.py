import pandas as pd
import csv
import glob
import os

for split in ["test"]:
    train_corpus = "phoenix14/%s.corpus.csv"%split

    base_path = "/home/panxie/workspace/sign-lang/fullFrame-210x260px/%s"%split

    train_df = pd.read_csv(train_corpus, sep='|')
    video_paths = train_df["folder"]

    with open("phoenix14/%s_split.csv"%split, "w") as f:
        writer = csv.writer(f, delimiter=',')
        for row in video_paths:
            vpath = os.path.join(base_path, row[:-5])
            content = [vpath, len(glob.glob(os.path.join(base_path, row)))]
            print(content)
            if row:
                writer.writerow(content)
        print('%s saved!'%split)