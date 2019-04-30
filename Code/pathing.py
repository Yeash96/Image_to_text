import os 

def main():
#    path = "..\\Input_image"

#    os.chdir(path)
#    cwd = os.getcwd()
#    print(cwd)
    
#    directory = os.listdir(path)
#    print(directory)

    path = '..\\Input_image\\Image001.jpg'
    print(os.path.split(path))
    print(os.path.isfile(path))


if __name__ == "__main__":
    main()
