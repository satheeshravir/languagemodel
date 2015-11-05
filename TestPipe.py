#Test pipe

import subprocess



subprocess.Popen(["python", "WithoutTokenizer.py", "tamil_corpus.txt", "test-ta.txt"])
#subprocess.Popen(["python", "WithTokenizer.py", "tamil_corpus.txt", "test-ta.txt"])
#subprocess.Popen(["python", "WithoutTokenizer.py", "demo-script-linked-q-a.txt", "test-en.txt"])
#subprocess.Popen(["python", "WithTokenizer.py", "demo-script-linked-q-a.txt", "en-test.txt"])
