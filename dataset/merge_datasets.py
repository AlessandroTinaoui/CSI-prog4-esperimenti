import os
import re
import glob

def main():
    pattern = "dataset_user_*_train.csv"
    files = glob.glob(pattern)

    if not files:
        print("Nessun file trovato con pattern:", pattern)
        return

    # Estrae n da dataset_user_n_train.csv per ordinare
    def extract_n(filename):
        base = os.path.basename(filename)
        m = re.match(r"dataset_user_(\d+)_train\.csv", base)
        if m:
            return int(m.group(1))
        return 10**9

    files_sorted = sorted(files, key=extract_n)

    print("Unisco i file nell'ordine:")
    for f in files_sorted:
        print(" -", f)

    output_file = "dataset_user_all_train.csv"

    first_file = True
    with open(output_file, "w", encoding="utf-8", newline="") as out_f:
        for f in files_sorted:
            with open(f, "r", encoding="utf-8", newline="") as in_f:
                for i, line in enumerate(in_f):
                    if i == 0:
                        # header: lo scrivo solo per il primo file
                        if first_file:
                            out_f.write(line)
                    else:
                        # tutte le altre righe sempre scritte
                        out_f.write(line)
            first_file = False

    print(f"\nFile unito salvato come: {output_file}")

if __name__ == "__main__":
    main()
