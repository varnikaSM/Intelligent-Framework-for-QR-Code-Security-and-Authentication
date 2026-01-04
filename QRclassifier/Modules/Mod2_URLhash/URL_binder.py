import hashlib
from urllib.parse import urlencode, urlparse, parse_qsl

#Normalising the URL to get it to lower case. For Eg: https://example.com/ is the same as https://Example.com, but both give different URL hashes if not normalised.
#The query parameters are also sorted.
def normalizeUrl(url: str) -> str:
    url=url.strip()
    parsed=urlparse(url)
    scheme=parsed.scheme.lower()
    netloc=parsed.netloc.lower()
    path=parsed.path.rstrip('/')
    if not path:
        path=''
    queryParameters=parse_qsl(parsed.query)
    queryParameters.sort()
    sortedQuery=urlencode(queryParameters)
    normalisedURL=f"{scheme}://{netloc}{path}"
    if sortedQuery:
        normalisedURL+=f"?{sortedQuery}"
    return normalisedURL

#SHA-256: Input: URL, Output: 32 byte (256 bit) binary data - unreadable by default
def hashURL(URL): 
    normalisedURL=normalizeUrl(URL)
    hashBytes=hashlib.sha256(normalisedURL.encode("utf-8")).digest() # digest is required by the machine, hexdigest is for human intervention
    return hashBytes

#hash to bitstream
def hashToBitstream(hashBytes):
    return "".join(format(byte, '08b') for byte in hashBytes) #08b: binary number, padded to 8 bits.

#Module2 output API
def generateURLid(URL):
    hashBytes=hashURL(URL)
    hashBitstream=hashToBitstream(hashBytes)
    return hashBitstream

#testing
urlOriginal = "https://college.edu/pay?id=101&user=admin"
urlSame = "https://college.edu/pay?user=admin&id=101"
urlAttack = "https://college.edu/Pay"

bitsOriginal = generateURLid(urlOriginal)
bitsSame = generateURLid(urlSame)
bitsAttack = generateURLid(urlAttack)
print("Original vs Same URL Match:", bitsOriginal == bitsSame)
print("Original vs Attack URL Match:", bitsOriginal == bitsAttack)
print("Bitstream Length:", len(bitsOriginal))
