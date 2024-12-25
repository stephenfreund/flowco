class YesNoPrompt:

    def __init__(self, yes_to_all: bool = False):
        self.yes_to_all = yes_to_all

    def ask(self, message: str) -> bool:

        if self.yes_to_all:
            print(f"  {message}? Answering yes to all.")
            return True

        while True:
            choice = input(f"  {message}? (y/Y/n): ").strip()
            if choice in ["y", "Y", "n"]:
                if choice == "Y":
                    self.yes_to_all = True
                    return True
                else:
                    return choice == "y"
            else:
                print(
                    "Please enter 'y' to accept, 'Y' to accept all, or 'n' to reject."
                )
