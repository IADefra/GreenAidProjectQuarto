import subprocess

def generate_requirements():
    try:
        # Specify the full path to pip
        pip_path = "C:\\Program Files\\Python37\\Scripts\\pip.exe"
        
        result = subprocess.run([pip_path, "freeze"], stdout=subprocess.PIPE, check=True, text=True)
        
        with open("requirements.txt", "w") as f:
            f.write(result.stdout)
        
        print("requirements.txt file has been generated successfully.")
    except subprocess.CalledProcessError as e:
        print("An error occurred while generating the requirements.txt file.")
        print(e.output)

if __name__ == "__main__":
    generate_requirements()


