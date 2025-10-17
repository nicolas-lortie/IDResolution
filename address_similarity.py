from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from functools import lru_cache
import re

def address_similarity(adr_1, adr_2, weight_fsa=0.2, weight_numbers=0.3, weight_Wratio=0.25, weight_JW=0.25, verbose=False):
    """
    Computes the similarity score between two addresses based on multiple factors.

    Args:
        adr_1 (str): The first address to compare.
        adr_2 (str): The second address to compare.
        weight_fsa (float, optional): Weight for the Forward Sortation Area (FSA) similarity. Default is 0.2.
        weight_numbers (float, optional): Weight for the similarity of numbers in the addresses. Default is 0.3.
        weight_Wratio (float, optional): Weight for the WRatio similarity score. Default is 0.25.
        weight_JW (float, optional): Weight for the Jaro-Winkler similarity score. Default is 0.25.
        verbose (bool, optional): If True, returns detailed scores as a dictionary. If False, returns only the overall score. Default is False.

    Returns:
        float or dict: 
            - If `verbose` is False, returns the overall similarity score as a float.
            - If `verbose` is True, returns a dictionary with individual scores and the overall score.
    
    Note: The weights should sum to 1.0 for meaningful results.
    """
    parsed_adr_1 = parse_address(adr_1)
    parsed_adr_2 = parse_address(adr_2)
    if parsed_adr_1 and parsed_adr_2:
        fsa_score = 1.0 if parsed_adr_1['fsa'] == parsed_adr_2['fsa'] and parsed_adr_1['fsa'] is not None else 0.0
        numbers_score = len(parsed_adr_1['numbers'].intersection(parsed_adr_2['numbers'])) / max(len(parsed_adr_1['numbers'].union(parsed_adr_2['numbers'])), 1)
        address_Wratio = fuzz.WRatio(parsed_adr_1['address'], parsed_adr_2['address']) / 100.0
        address_JW = JaroWinkler.normalized_similarity(parsed_adr_1['address'], parsed_adr_2['address'])
        
        overall_score = (weight_fsa * fsa_score) + (weight_numbers * numbers_score) + (weight_Wratio * address_Wratio) + (weight_JW * address_JW)
        
        if verbose:
            return {
                "fsa_score": fsa_score,
                "numbers_score": numbers_score,
                "address_Wratio": address_Wratio,
                "address_JW": address_JW,
                "overall_score": overall_score
            }
        else:
            return overall_score
 
@lru_cache(maxsize=1000)
def parse_address(address):
    """
    Parses an address to extract the postal code, FSA, and numbers.
    Args:
        address (str): The address to parse.
    Returns:
        dict: A dictionary containing the parsed address components.
    """
    parsed_address = {}
    parsed_address['address'] = address
    parsed_address['postal_code'] = re.search(r'\b([A-Z]\d[A-Z]\s?\d[A-Z]\d)\b', address).group(0) if re.search(r'\b([A-Z]\d[A-Z]\s?\d[A-Z]\d)\b', address) else None
    parsed_address['fsa'] = parsed_address['postal_code'][:3] if parsed_address['postal_code'] else None
    parsed_address['numbers'] = set(re.findall(r'\b\d+[A-Za-z]?\b', address))

    return parsed_address




