import shutil
import os
import subprocess
import math

from user_config import DATA_PATH, ENV_PATH

applications = ["plants", "acmeair", "daytrader", "petclinic-legacy", "roller"]
sizes = [40, 86, 118, 60, 531, 100, 43, 119]
job_filename = "cc_job.sh"
prefix_command = ""
for app, size in zip(applications, sizes):
    if not os.path.exists(os.path.join(os.path.curdir, "logs", app.lower())):
        os.mkdir(os.path.join(os.path.curdir, "logs", app.lower()))
    new_job_filename = f"{job_filename[:-3]}_{app.lower()}{job_filename[-3:]}"
    shutil.copy(os.path.join(os.path.curdir, "cc_scripts", job_filename),
                os.path.join(os.path.curdir, new_job_filename))
    center = int(math.ceil(math.sqrt(size)))
    radius = int(math.ceil(math.sqrt(size)/3))
    min_mnm = center - radius
    max_mnm = center + radius

    with open(os.path.join(os.path.curdir, new_job_filename), "r") as f:
        lines = [line.replace("${ENV_PATH}", ENV_PATH).replace("${MAX_MNM}", str(max_mnm)).replace("${MIN_MNM}",
            str(min_mnm)).replace("${DATA_PATH}", DATA_PATH).replace("$APP", app.lower()) for line in f.readlines()]
    with open(os.path.join(os.path.curdir, new_job_filename), "w") as f:
        f.writelines(lines)
    subprocess.call(['sbatch', new_job_filename])