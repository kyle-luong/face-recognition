from enroll import enroll_face
from verify import verify_face

def main():
    print("Choose an option:")
    print("1. Enroll a new face")
    print("2. Verify faces")
    choice = input("Enter your choice (1/2): ")

    if choice == "1":
        name = input("Enter the name of the person to enroll: ")
        enroll_face(name)
    elif choice == "2":
        verify_face()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()