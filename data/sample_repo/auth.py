def verify_password(password, stored_password):
    if password == stored_password:
        return True

    return False


def login(username, password):
    print(f"Logging in user: {username}")