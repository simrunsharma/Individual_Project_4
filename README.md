# Auto-Scaling Container Project: Individual Project 4
[![CI](https://github.com/simrunsharma/Individual_Project_4/actions/workflows/cicd.yml/badge.svg)](https://github.com/simrunsharma/Individual_Project_4/actions/workflows/cicd.yml)

Youtube Link: https://youtu.be/N_WTppKuK4E

## Overview

For this assignment, you will be creating a publicly accessible auto-scaling container using Azure App Services and Flask. This project provides an opportunity to apply your Flask knowledge from our Data Engineering Class and gain experience with building and deploying scalable web-hosted applications.

## Flask App (Generative Sentence Completion App)
The Flask application utilizes the Hugging Face Transformers library to implement a text summarization tool powered by the GPT-2 language model. Users input text through a form on the homepage ('/'), and the application, upon a POST request to '/generate', encodes the input using GPT-2 tokenizer, generates a summary with the model, and displays the original input alongside the generated text on the 'result.html' page. The GPT-2 model, loaded from Hugging Face's Transformers, is fine-tuned for language modeling, ensuring coherent and contextually relevant text generation. The app runs in debug mode, making it accessible at 'http://localhost:5000/'. 
- This link only works when you run the command python app.py as it starts a development server and makes the specified port (by default, it's usually 5000) accessible on your local machine.
- This repository contains a simple web application built for generative sentence completion. The application utilizes Flask, a Python web framework, to create an interface for users to input text and receive an autocompleted sentence generated by a language model.

## Files:

- **app.py**: The main Python script that defines the Flask application. It handles incoming requests, processes user input, and generates autocompleted sentences using a pre-trained language model.

- **index.html**: The HTML template for the user interface. It provides a form where users can input text for summarization.

- **result.html**: Another HTML template that displays the original input text and the autocompleted sentence generated by the application.

## Usage:

1. **Input Text**: Users can enter a text snippet into the provided textarea on the home page.

2. **Generate Summary**: Clicking the "Generate Summary" button sends the user's input to the server, which processes the text and generates an autocompleted sentence.

3. **View Result**: The result page displays both the original input text and the autocompleted sentence, providing a convenient way for users to compare the input and output.

## Styling:

- The user interface is designed with a clean and user-friendly layout using HTML and CSS. The form elements are styled for readability and ease of use.

## Deployment:

The application can be deployed on a server to make it publicly accessible. The Flask web framework handles the server-side logic, allowing users to interact with the application through a web browser.

Please note that the provided HTML files focus on the user interface, and the primary logic for autocompletion resides in the Python script (app.py).


### Fully Functioning Embedded LLM within Flask: GPT-2 Model
- Here is the UI Interface of my app:
![Model](https://private-user-images.githubusercontent.com/141798228/289319591-bf508209-e731-4728-b7cb-f954ccdc0b3c.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDIxNjU4MzIsIm5iZiI6MTcwMjE2NTUzMiwicGF0aCI6Ii8xNDE3OTgyMjgvMjg5MzE5NTkxLWJmNTA4MjA5LWU3MzEtNDcyOC1iN2NiLWY5NTRjY2RjMGIzYy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMjA5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTIwOVQyMzQ1MzJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT01OWM0MDlkNDJhZTEzOTdjN2Y3MjRlOWJhM2VmMGY2MmRhNjllNGE2YzZmNjE5NjhkNzgwMTg3ZjI1MzgyZjJjJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.6KP9BxljIXEqSh3U-6byM292HlINf12R85IJu9ALxdA)
- This is the generated output of the model:
  ![GeneratedSummary](https://private-user-images.githubusercontent.com/141798228/289319882-cd2b465c-f5bd-492a-bf4e-30f469ae5b33.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDIxNjU5MzcsIm5iZiI6MTcwMjE2NTYzNywicGF0aCI6Ii8xNDE3OTgyMjgvMjg5MzE5ODgyLWNkMmI0NjVjLWY1YmQtNDkyYS1iZjRlLTMwZjQ2OWFlNWIzMy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMjA5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTIwOVQyMzQ3MTdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1kNTY4YmZmNjYyMDNhNzExZGRjNTdjMGNlYTkwZmRmYzliMjQwMjEwZWJhODM3YWFiZDE5N2JiYjBhYmZjMTg1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.R-3t1UCKfSOKNMHbkTSinUHQvFZj4Q2cgnKFHnMaA1s)


### First Create A DockerHub Account: 

- If you click on this [Link](https://hub.docker.com/signup) to signup for DockerHub.
- You click on Create Repository
- Select a Name and Description for the Repository.
- Public or Private Repository Visibility.
- Create

### Once you Create DockerHub you are given this code:
- This code is important to push docker images to DockerHub: You can push a new image to this repository using the CLI
  ```
  docker tag local-image:tagname new-repo:tagname
  ```
  ```
  docker push new-repo:tagname
  ```
- Here is a representation of what the image looks like in DockerHub:
  ![DockerHub](https://private-user-images.githubusercontent.com/141798228/289317857-24210a70-9351-40c4-8567-8cfed9db9424.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDIxNjQ3MzMsIm5iZiI6MTcwMjE2NDQzMywicGF0aCI6Ii8xNDE3OTgyMjgvMjg5MzE3ODU3LTI0MjEwYTcwLTkzNTEtNDBjNC04NTY3LThjZmVkOWRiOTQyNC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMjA5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTIwOVQyMzI3MTNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT01ZjFhOTQ3NWRmMmQzODFiNTQ4YjViMWRmY2Q2YTIyNjQwZTdhYjNmNTVmZmU3ZDM0OTMxZjQ5OTgxMmRmOGQwJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.QbY_yv3mBT26AKzrbfwbzlXUAvnT5NnSQxQkMUZPU54)

### Dockerfile:
- This DockerFile is different from the one in the dev container as it is specific to my app that I am trying to containerize.
  ![Dockerfile Issue](https://private-user-images.githubusercontent.com/141798228/289317528-04664e46-80a5-4f5c-8656-88dcd8def1d7.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDIxNjQ0NDAsIm5iZiI6MTcwMjE2NDE0MCwicGF0aCI6Ii8xNDE3OTgyMjgvMjg5MzE3NTI4LTA0NjY0ZTQ2LTgwYTUtNGY1Yy04NjU2LTg4ZGNkOGRlZjFkNy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMjA5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTIwOVQyMzIyMjBaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1lMGZkMThkMWMwOWRkMzAwNjMxNjZmZWQ2ZGUzZmU1OTVhOGE5YTgwZGY4MjMyZjcyNDZiMzNiODc1NDMwODE3JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.Cgg49iR4t1ll5HcmBrzB8gp53KGgq_D7K81VZJh7BOE)

## Docker Image Creation:
Here is a list of commands to create a docker image:
1. ``` docker build -t dockerusername/imagetag . ```
2. ```docker images ```
3. ``` docker push dockerusername/imagetag ```
4. ``` docker login -u dockerusername -p dockerpassword ```

Here is a representation what the image looks like from the terminal: 
![Docker Image CLI](https://private-user-images.githubusercontent.com/141798228/289317888-eacbc468-0134-46a6-a541-9771fe38e451.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDIxNjQ3NjQsIm5iZiI6MTcwMjE2NDQ2NCwicGF0aCI6Ii8xNDE3OTgyMjgvMjg5MzE3ODg4LWVhY2JjNDY4LTAxMzQtNDZhNi1hNTQxLTk3NzFmZTM4ZTQ1MS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMjA5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTIwOVQyMzI3NDRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT04ZGRlOGE5YmUzYmEyYTJjM2UzOGMxMjM4NTgxMjZkMDgxODIzOTgyOGZiNzczYTk3OWY2ZjlhMjlmMWZiNGRlJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.Rhz7gfsqMivtzhHQz7UHA8D4eMASaPNiCyCZ_uT8QC4)

Here is a representation of the image in Dockerhub specifics: 
![DockerImage](https://private-user-images.githubusercontent.com/141798228/289317981-fcbf55e9-4c3d-4d0e-bc79-3188647f0d4b.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDIxNjQ4MzUsIm5iZiI6MTcwMjE2NDUzNSwicGF0aCI6Ii8xNDE3OTgyMjgvMjg5MzE3OTgxLWZjYmY1NWU5LTRjM2QtNGQwZS1iYzc5LTMxODg2NDdmMGQ0Yi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMjA5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTIwOVQyMzI4NTVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0zMDZiNGY0ZTcxYjM0MGIxZWY4OWY1ZjhjZmE2NTQzZjRjODg4NTBlYTlhMmZkYmU0OTBkZTg5ZThkNWExNTdiJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.PH5um6H6Dm8KndVvzvgLjIjjoUGvRm7pWXIKa3ZetX0)


## Azure Web App and Using Docker:
- First you must go to this site: [CreateWebApp](https://portal.azure.com/#create/Microsoft.WebSite).
- Once there you need to create a resource group which should be aligned to the project or web app you are creating
- Name of the instance should be the name of the web app name of your website
- Publish using DOCKER CONTAINER
- Operating System is Linux
- Region should be East US but if you run into the issue that scheduled features are not working use Japan East.
- Then go to the DOCKER tab - single container and use the repository/image tag
- Create and Review

  Here is a representation of these steps:
  ![Creating Azure Web App](https://private-user-images.githubusercontent.com/141798228/289318033-8fb57265-6dbe-4335-903b-343dac0f94e9.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDIxNjQ5MDYsIm5iZiI6MTcwMjE2NDYwNiwicGF0aCI6Ii8xNDE3OTgyMjgvMjg5MzE4MDMzLThmYjU3MjY1LTZkYmUtNDMzNS05MDNiLTM0M2RhYzBmOTRlOS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMjA5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTIwOVQyMzMwMDZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1iNWM0N2YzZjY1MTBjNzY4ZjE3Yzg5NDYzNWE3MzM1NmNhOTU3ZDI3ZDZlODZhZDE0ODZhNDk2ZDlmZWYwYWU5JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.ZMDqjhQbIvMWN2t1swShUHy7Ls6H-I-ogvdONVLsItI)

Here is a representation of what the web app looks like once deployed:(mine is currently stopped to save money):
![Web Product](https://private-user-images.githubusercontent.com/141798228/289325001-cab7cfed-2d1b-4d5c-a10d-2e0f13c794ac.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDIxNjgwMDYsIm5iZiI6MTcwMjE2NzcwNiwicGF0aCI6Ii8xNDE3OTgyMjgvMjg5MzI1MDAxLWNhYjdjZmVkLTJkMWItNGQ1Yy1hMTBkLTJlMGYxM2M3OTRhYy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMjEwJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTIxMFQwMDIxNDZaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1jMGRiZTMwNTU5NzUyYTVkMjg0OTk2ZDlmNDZkYmEwNzM4YWFiOTU5ZGNhZDZiZDRhMGVlMzI0YTlmMjE4MDkyJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.avXrWULxw-CAZF_q2TEpNq00ilGabYwB8CZnlomF4jw)

