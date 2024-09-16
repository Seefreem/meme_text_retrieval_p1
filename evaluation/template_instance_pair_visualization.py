import inspect
import json
import argparse

def create_html(test_memes_configs):
    print(inspect.currentframe().f_code.co_name)
    image_sets = []
    print(f'In total, there are {len(test_memes_configs)} pairs')
    for ite in test_memes_configs:
        file_path = ite["template_file_name"][0]
        image_sets.append({"images": [ite["meme_name"], file_path]})



    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Sets Table</title>
        <style>
            table {
                width: 100%;
                border-collapse: collapse;
            }
            td {
                padding: 10px;
                text-align: center;
                border: 1px solid #ddd;
            }
            img {
                max-width: 100%;
                height: auto;
            }
            p {
                margin-top: 5px;
                font-size: 14px;
                color: #555;
            }
        </style>
    </head>
    <body>
        <h1>Image Sets</h1>
        <table>
    '''
    # Add rows for each image set
    for image_set in image_sets:
        html_content += '<tr>'
        for image in image_set['images']:
            # Add the image along with its filename
            html_content += f'''
            <td>
                <img src="{image}" alt="Image">
                <p>{image}</p>
            </td>
            '''
        html_content += '</tr>'


    # Close the HTML tags
    html_content += '''
        </table>
    </body>
    </html>
    '''

    # Save the HTML to a file or print it out
    with open('image_sets.html', 'w') as f:
        f.write(html_content)

    print("HTML file generated: image_sets.html")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='meme pair visualization')
    parser.add_argument('--filename', action='store', type=str, dest='filename', default='meme2template.json')
    args = parser.parse_args()

    test_memes_configs = []
    with open(args.filename, 'r', encoding='utf-8') as json_file:
            test_memes_configs = json.load(json_file)
    
    create_html(test_memes_configs)
