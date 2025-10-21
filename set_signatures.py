import mmh3 
import numpy as np




# 2^61 - 1 is a common large prime for 64-bit hashing arithmetic
_PRIME = np.uint64((1 << 61) - 1)
_MAX_HASH = np.uint64((1 << 32) - 1)



class SetSignatures:
     
    def __init__(self,
                   num_perm : int,
                   seed : int = 42,
                   shingle_size : int = 3,
                   treshold: float = 0.7
                   ):
        self.num_perm = num_perm
        self.seed = seed
        self.shingle_size = shingle_size
        self.treshold = treshold
        self._a, self._b = self._permutations()
        self.num_bands, self.rows_per_band, self.actual_treshold = self._pick_b_r(self.num_perm, self.treshold)
    
    def _permutations(self):
        rng = np.random.default_rng(self.seed)
        # a in [1, p-1], b in [0, p-1]
        a = rng.integers(1, _PRIME, size=self.num_perm, dtype=np.uint64)
        b = rng.integers(0, _PRIME, size=self.num_perm, dtype=np.uint64)
        return a, b
    
    def _pick_b_r(self, num_perm, threshold):
        candidates = []
        for r in range(1, num_perm + 1):
            if num_perm % r != 0:
                continue
            b = num_perm // r
            s = (1 / b) ** (1 / r)
            candidates.append((b, r, s, abs(s - threshold)))
        return min(candidates, key=lambda x: x[3])[:3]


    def _compute_minhash(self, shingles):
        """
        tokens: iterable of shingles (strings or bytes)
        returns: np.ndarray shape (num_perm,) dtype=uint64 (ordered signature)
        """

        basehash = np.fromiter(
            (mmh3.hash64(shingle, seed=self.seed, signed=False)[0] for shingle in shingles),
            dtype=np.uint64
        ) 

        basehash = (basehash % _PRIME).astype(np.uint64)     
        
        basehash = basehash[:, None].astype(np.uint64)                
        a  = self._a[None, :]                                 
        b  = self._b[None, :]                                   

        permuted_hashes = (basehash * a + b) % _PRIME & _MAX_HASH           

        minhash_signature = np.minimum.reduce(permuted_hashes, axis=0).astype(np.uint64)  
        return minhash_signature
    
    def _compute_lsh_key_int(self, minhash_signature: np.ndarray) -> list[int]:
        sig = np.ascontiguousarray(minhash_signature.astype(np.dtype('<u8'), copy=False))
        out = []
        for band in range(self.num_bands):
            s = band * self.rows_per_band
            e = s + self.rows_per_band
            bucket64, _ = mmh3.hash64(sig[s:e].tobytes(), seed=self.seed, signed=False)
            out.append((band << 64) | bucket64)  
        return out
    
    def get_lsh_key(self, text):
        shingles = self.shingles(text)
        minhash_signature = self._compute_minhash(shingles)
        lsh_keys = self._compute_lsh_key_int(minhash_signature)
        return lsh_keys
    
    def get_batch_lsh_key(self, texts: list[str]) -> list[list[int]]:
        all_keys = []
        for text in texts:
            shingles = self.shingles(text)
            minhash_signature = self._compute_minhash(shingles)
            lsh_keys = self._compute_lsh_key_int(minhash_signature)
            all_keys.append(lsh_keys)
        return all_keys 


    def shingles(self, text: str) -> set:
            """Generate shingles of given size from the input text."""
            return {text[i:i + self.shingle_size] for i in range(len(text) - self.shingle_size + 1)}



