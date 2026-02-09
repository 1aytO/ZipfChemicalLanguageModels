import os
import re
import csv
import time
import json
import array
import math
import platform
from collections import Counter
from pathlib import Path
from multiprocessing import Process, Pipe, cpu_count

# -------------------------
# Config
# -------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "chembl_selfies.txt"
OUT_DIR   = REPO_ROOT / "outputs" / "bpe_stats_sharded"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_MERGES = 4000

WORKER_COUNT = max(1, cpu_count() - 1) 

os.makedirs(OUT_DIR, exist_ok=True)
STEP_LOG_PATH = os.path.join(OUT_DIR, "step_log.csv")
FINAL_VOCAB_PATH = os.path.join(OUT_DIR, "final_vocab.json")
MERGES_PATH = os.path.join(OUT_DIR, "merges.txt")

# -------------------------
# Helpers
# -------------------------
SELFIES_TOKEN_RE = re.compile(r"\[[^\]]+\]")

def enc_pair(a, b):
    return (int(a) << 32) | int(b)

def dec_pair(x):
    return (int(x >> 32), int(x & 0xFFFFFFFF))

# -------------------------
# Worker Process Class
# -------------------------
class BPEWorker(Process):
    def __init__(self, worker_id, pipe, data_lines, initial_vocab):
        super().__init__()
        self.worker_id = worker_id
        self.pipe = pipe
        self.lines = data_lines
        self.token2id = initial_vocab.copy()
        

        self.tokens = array.array("I")
        self.offsets = array.array("Q", [0])
        self.active_len = 0 

    def run(self):
        local_toks = []
        current_offset = 0
        
        for line in self.lines:
            s = line.strip()
            if not s: continue
            toks = SELFIES_TOKEN_RE.findall(s)
            
            for t in toks:
                if t not in self.token2id:
                    pass 
                local_toks.append(self.token2id[t])
            
            current_offset += len(toks)
            self.offsets.append(current_offset)
        
        self.tokens = array.array("I", local_toks)
        self.active_len = len(self.tokens)
        

        self.lines = None 
        

        pc = self.count_pairs()
        self.pipe.send(("READY", pc, self.active_len))


        while True:
            cmd, args = self.pipe.recv()
            
            if cmd == "MERGE":
                a, b, new_id = args
                delta, occ, removed_cnt = self.apply_merge(a, b, new_id)
                self.active_len -= removed_cnt
                self.pipe.send(("OK", delta, occ, self.active_len))
            
            elif cmd == "GET_VOCAB":
                final_counts = Counter(self.tokens)
                self.pipe.send(final_counts)
                break
                
            elif cmd == "STOP":
                break

    def count_pairs(self):
        pc = Counter()
        tokens = self.tokens
        offsets = self.offsets
        n_mols = len(offsets) - 1
        
        for i in range(n_mols):
            s = offsets[i]
            e = offsets[i+1]
            if e - s < 2: continue
            for j in range(s, e - 1):
                pc[enc_pair(tokens[j], tokens[j+1])] += 1
        return pc

    def apply_merge(self, a, b, new_id):
        tokens = self.tokens
        offsets = self.offsets
        n_mols = len(offsets) - 1
        
        a = int(a); b = int(b); new_id = int(new_id)
        
        delta = Counter()
        total_occ = 0
        removed_tokens_cnt = 0
        

        new_tokens_arr = array.array("I")
        new_offsets = array.array("Q", [0])
        curr_off = 0
        
        for i in range(n_mols):
            s = offsets[i]
            e = offsets[i+1]
            length = e - s
            
            if length < 2:
                # 没变化
                for k in range(s, e):
                    new_tokens_arr.append(tokens[k])
                curr_off += length
                new_offsets.append(curr_off)
                continue
            

            prev_new = None
            mol_toks = []
            

            merged_indices = []
            k = s
            while k < e - 1:
                if tokens[k] == a and tokens[k+1] == b:
                    merged_indices.append(k)
                    k += 2
                else:
                    k += 1
            
            if not merged_indices:
                for k in range(s, e):
                    new_tokens_arr.append(tokens[k])
                curr_off += length
                new_offsets.append(curr_off)
                continue
            

            total_occ += len(merged_indices)
            removed_tokens_cnt += len(merged_indices)

            for k in range(s, e - 1):
                delta[enc_pair(tokens[k], tokens[k+1])] -= 1

            k = s
            temp_mol = []
            while k < e:
                if k < e - 1 and tokens[k] == a and tokens[k+1] == b:
                    temp_mol.append(new_id)
                    k += 2
                else:
                    temp_mol.append(tokens[k])
                    k += 1

            for t in temp_mol:
                new_tokens_arr.append(t)

            for k in range(len(temp_mol) - 1):
                delta[enc_pair(temp_mol[k], temp_mol[k+1])] += 1
                
            curr_off += len(temp_mol)
            new_offsets.append(curr_off)

        self.tokens = new_tokens_arr
        self.offsets = new_offsets
        
        return delta, total_occ, removed_tokens_cnt

# -------------------------
# Main Controller
# -------------------------
def main():
    print(f"Starting Fast BPE with {WORKER_COUNT} persistent workers...")
    

    all_lines = []
    print("Reading file...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    
    total_lines = len(all_lines)
    chunk_size = math.ceil(total_lines / WORKER_COUNT)
    print(f"Total molecules: {total_lines:,}. Split into {WORKER_COUNT} chunks.")

    print("Building initial vocab...")
    token2id = {"[*]": 0}
    id2token = ["[*]"]

    temp_re = re.compile(r"\[[^\]]+\]")
    for line in all_lines:
        for t in temp_re.findall(line):
            if t not in token2id:
                token2id[t] = len(id2token)
                id2token.append(t)
    
    initial_tokens_count = 0

    workers = []
    pipes = []
    
    for i in range(WORKER_COUNT):
        p_conn, c_conn = Pipe()
        start = i * chunk_size
        end = start + chunk_size
        chunk_lines = all_lines[start:end]
        
        w = BPEWorker(i, c_conn, chunk_lines, token2id)
        w.start()
        workers.append(w)
        pipes.append(p_conn)
    
    del all_lines
    

    global_pair_counts = Counter()
    total_tokens = 0
    
    print("Waiting for workers to initialize...")
    for p in pipes:
        msg, pc, t_len = p.recv()
        global_pair_counts.update(pc)
        total_tokens += t_len
        
    initial_tokens_count = total_tokens
    print(f"Initialization done. Total tokens: {total_tokens:,}. Unique pairs: {len(global_pair_counts):,}")

    with open(STEP_LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "pair", "freq", "total_tokens", "compression", "time_sec"])

    merges_buffer = []
    t_start = time.time()

    for step in range(1, NUM_MERGES + 1):
        step_t0 = time.time()
        
        if not global_pair_counts:
            break

        best = global_pair_counts.most_common(1)
        if not best: break
        
        best_enc, best_freq = best[0]
        if best_freq <= 0:
            global_pair_counts = +global_pair_counts
            best = global_pair_counts.most_common(1)
            if not best or best[0][1] <= 0: break
            best_enc, best_freq = best[0]
            
        a, b = dec_pair(best_enc)
        

        new_token_str = id2token[a] + id2token[b]
        new_id = len(id2token)
        id2token.append(new_token_str)
        token2id[new_token_str] = new_id
        merges_buffer.append(f"{id2token[a]} {id2token[b]}")
        

        for p in pipes:
            p.send(("MERGE", (a, b, new_id)))
            

        step_occ = 0
        step_removed = 0
        
        for p in pipes:
            msg, delta, occ, active_len = p.recv()
            global_pair_counts.update(delta)
            step_occ += occ
            step_removed += occ
            

        del global_pair_counts[best_enc]
        total_tokens -= step_removed
        
        step_t1 = time.time()
        

        if step % 1 == 0 or step == 1:
            print(f"Step {step}/{NUM_MERGES} | Merge {id2token[a]}+{id2token[b]} | Freq {best_freq} | Toks {total_tokens} | Time {step_t1-step_t0:.3f}s")
            
            with open(STEP_LOG_PATH, "a", newline="") as f:
                csv.writer(f).writerow([
                    step, f"{id2token[a]}+{id2token[b]}", best_freq, 
                    total_tokens, f"{initial_tokens_count/total_tokens:.4f}", 
                    f"{time.time()-t_start:.1f}"
                ])

    print("Training finished. Collecting final stats...")
    

    for p in pipes:
        p.send(("GET_VOCAB", None))
        

    final_vocab_counts = Counter()
    for p in pipes:
        c = p.recv()
        final_vocab_counts.update(c)
        
    for w in workers:
        w.join()


    final_vocab_dict = {id2token[tok_id]: cnt for tok_id, cnt in final_vocab_counts.items()}
    with open(FINAL_VOCAB_PATH, "w") as f:
        json.dump(final_vocab_dict, f, indent=2)
        
    with open(MERGES_PATH, "w") as f:
        f.write("\n".join(merges_buffer))
        
    print(f"Done. Saved to {OUT_DIR}")

if __name__ == "__main__":
    main()