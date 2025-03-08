import os
import requests
from bs4 import BeautifulSoup
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

name = "SN2024fa"
base_url = "https://spt3g.ncsa.illinois.edu/files/jobs/d4193105e56c447b8d94d61d01bcfb57/out/SN2024fa/"

download_folder = f"/Volumes/Memorex USB/{name}/fits_files"
os.makedirs(download_folder, exist_ok=True)


response = requests.get(base_url)
response.raise_for_status()
soup = BeautifulSoup(response.text, "html.parser")

file_links = [
    base_url + link["href"]
    for link in soup.find_all("a", href=True)
    if not link["href"].startswith("?") and not link["href"].endswith("/")
]

print(f"Found {len(file_links)} files to download.")

for idx, file_url in enumerate(file_links, start=1):
    file_name = os.path.basename(file_url)
    save_path = os.path.join(download_folder, file_name)

    try:
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"({idx}/{len(file_links)}) Downloaded: {file_name}")
    except Exception as e:
        print(f"Error downloading {file_name}: {e}")


input_folder = download_folder
output_csv = "/Users/Djslime07/SPT_VSCode/CSV's/obs_values.csv"

for fits_file in os.listdir(input_folder):
    if fits_file.endswith(".fits"):
        fits_path = os.path.join(input_folder, fits_file)

        if fits_file.startswith("._"):
            continue

        try:
            with fits.open(fits_path, ignore_missing_simple=True) as hdul:
                data = hdul[0].data
                header = hdul[0].header
                date_beg = header.get('DATE-BEG', 'Unknown')
                date_end = header.get('DATE-END', 'Unknown')
                object_name = header.get('OBJECT', 'Unknown')
                band = header.get('BAND', 'Unknown')
                print(f"Processing {fits_file} | DATE-BEG: {date_beg} | BAND: {band}")

                try:
                    mjd = Time(date_beg).mjd
                except Exception as e:
                    print(f"Skipping {fits_file}: Invalid DATE-BEG format. Error: {e}")
                    continue

                if data is None:
                    print(f"Skipping {fits_file}: No image data found.")
                    continue

                data = np.nan_to_num(data)
                if np.max(data) == 0:
                    print(f"Skipping {fits_file}: Peak flux is zero.")
                    continue


                if "fltd" in fits_file:
                    suffix = "Filtered"
                elif "psth" in fits_file:
                    suffix = "Passthrough"
                else:
                    print(f"Skipping {fits_file}: Unknown file type.")
                    continue

                peak_flux = np.max(data)
                rms_temp = np.std(data)
                keep = np.abs(data) <= 3 * rms_temp
                rms = np.std(data[keep])
                vmin, vmax = np.percentile(data, [1, 99])


                band_output_folder = f"/Volumes/Memorex USB/{name}/PNG_{name}_{band}"
                os.makedirs(band_output_folder, exist_ok=True)

                plt.figure()
                plt.imshow(data, origin='lower', vmin=vmin, vmax=vmax)
                plt.colorbar(label='Flux [mK]')
                plt.title(
                    f"{object_name} ({band}) MJD: {mjd:.5f} | Peak Flux: {peak_flux:.4} mK | Flux RMS: {rms:.4f} mK",
                    fontsize=9.5
                )
                png_filename = os.path.join(band_output_folder, f"{mjd:.5f}_{suffix}.png")
                plt.savefig(png_filename)
                plt.close()


                with open(output_csv, "a") as f:
                    print(f"{object_name},{date_beg},{date_end},{mjd},{peak_flux},{rms},{band}", file=f)

            print(f"Converted: {fits_file} -> {png_filename}")
        except Exception as e:
            print(f"Error processing {fits_file}: {e}")