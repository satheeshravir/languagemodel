#Test pipe

import subprocess



#subprocess.Popen(["python", "WithoutTokenizer.py", "demo-script-linked-q-a.txt", "en-ta.txt"])
#subprocess.Popen(["python", "WithTokenizer.py", "demo-script-linked-q-a.txt", "en-ta.txt"])
#subprocess.Popen(["python", "WithoutTokenizer.py", "demo-script-linked-q-a.txt", "en-test.txt"])
subprocess.Popen(["python", "WithTokenizer.py", "demo-script-linked-q-a.txt", "en-test.txt"])
