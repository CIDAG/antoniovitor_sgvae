from pathlib import Path
import urllib.request
import zipfile

class IL_ESW_Datasource():
    datasource_path = Path('../datasources/IL_ESW')

    def download_IL_ESW_repository(self):
        url = 'https://www.dropbox.com/s/yfe8zfb50ubtse0/IL-ESW_dataset.zip?dl=1'

        self.datasource_path.mkdir(parents=True, exist_ok=True)

        # checking if repository was downloaded
        if((self.datasource_path / 'dataset_complete').exists()):
            return

        print('Downloading IL_ESW repository... ', end='')
        zip_path = self.datasource_path / 'IL_ESW.zip'
        urllib.request.urlretrieve(url, zip_path)
        print('Done')

        print('Extracting IL_ESW repository... ', end='')
        with zipfile.ZipFile(zip_path) as file:
            file.extractall(self.datasource_path)
        zip_path.unlink()
        print('Done')

class IL_ESW_Extended_Datasource():
    datasource_path = Path('../datasources/IL_ESW_extended')

    def download_IL_ESW_extended(self):
        url = 'https://www.dropbox.com/scl/fi/5oqs6iq5zdx5ztjomjv7v/ions_extended.zip?rlkey=a2hbi2v25f7lv6apz9ud6yd41&dl=1'
        self.datasource_path.mkdir(parents=True, exist_ok=True)

        # checking if repository was downloaded
        if((self.datasource_path / f'anions.csv').exists()): return

        print('Downloading IL_ESW_extended... ', end='')
        zip_path = self.datasource_path / 'IL_ESW_extended.zip'
        urllib.request.urlretrieve(url, zip_path)
        print('Done')

        print('Extracting IL_ESW_extended... ', end='')
        with zipfile.ZipFile(zip_path) as file:
            file.extractall(self.datasource_path)
        zip_path.unlink()
        print('Done')