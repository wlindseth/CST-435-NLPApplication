import nltk
import ssl

print("Attempting to download NLTK resources...")

try:
    # Create an unverified SSL context
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # This is for older Python versions
    pass
else:
    # This is the magic part: tell Python to use the unverified context
    ssl._create_default_https_context = _create_unverified_https_context

# Now, try to download the packages
try:
    nltk.download('stopwords')
    nltk.download('punkt') # Still good to have
    nltk.download('punkt_tab') # <-- This is the new one we need!

    print("\nSuccessfully downloaded 'stopwords', 'punkt', and 'punkt_tab'!")
except Exception as e:
    print(f"\nAn error occurred during download: {e}")
    print("Please check your internet connection.")