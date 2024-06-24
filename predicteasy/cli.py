import os
import sys
import requests
import typer
from typing_extensions import Annotated
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

app = typer.Typer()

load_dotenv()
worker_url = os.getenv('WORKER_SERVICES')

@app.command()
def login(interactive: Annotated[bool, typer.Option("--i")] = False):
    """
    Log in to PredictEasy CLI.
    """
    if interactive:
        email = typer.prompt("Email:")
        password = typer.prompt("Password:", hide_input=True)
        try:
            # Send login request
            response = requests.post(
                "https://accounts.cleverinsight.co/beta/auth/login",
                json={"email": email, "password": password}
            )
            response.raise_for_status()  
            auth_data = response.json() 

            if 'accessToken' not in auth_data:
                typer.echo("Access token not found in the response.")
                return
            
            accessToken = auth_data['accessToken']
            typer.echo(f"Logged in as {email}")

            # TODO: Save the access token for future requests, e.g., in a file or environment variable
            
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 400:
                typer.echo("Incorrect Email or Password! Try again!")
            else:
                typer.echo(f"HTTP error occurred: {http_err}")
        except requests.exceptions.ConnectionError:
            typer.echo("Connection error. Please check your internet connection.")
        except requests.exceptions.Timeout:
            typer.echo("Request timed out. Please try again later.")
        except requests.exceptions.RequestException as req_err:
            typer.echo(f"An error occurred: {req_err}")
        except KeyError as key_err:
            typer.echo(f"Missing expected key in response: {key_err}")
        except Exception as e:
            typer.echo(f"An unexpected error occurred: {e}")

# else: TODO: Implement login via Browser API if not interactive - After API created

if __name__ == "__main__":
    app()
