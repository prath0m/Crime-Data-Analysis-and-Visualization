# Crime-Data-Analysis-and-Visualization

## Installation

1. **Clone the Repository**: 
    ```bash
    git clone https://github.com/your-username/your-django-project.git
    ```
    Replace `your-username` with your GitHub username and `your-django-project` with the name of your Django project repository.

2. **Navigate to Project Directory**:
    ```bash
    cd your-django-project
    ```

3. **Install Python**: 
    Make sure you have Python installed on your system. You can download it from [here](https://www.python.org/downloads/).

4. **Create Virtual Environment** (optional but recommended):
    ```bash
    python -m venv env
    ```

5. **Activate Virtual Environment** (if created):
    - On Windows:
        ```bash
        .\env\Scripts\activate
        ```
    - On macOS and Linux:
        ```bash
        source env/bin/activate
        ```

6. **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1. **Database Configuration**:
    - Open `settings.py` in your project directory.
    - Update the `DATABASES` setting according to your database configuration.

2. **Static Files Configuration (Optional)**:
    - If you are serving static files (CSS, JavaScript, images) in production, make sure to configure the `STATIC_URL` and `STATIC_ROOT` settings in `settings.py`.

## Running the Project

1. **Migrate Database**:
    ```bash
    python manage.py migrate
    ```

2. **Create Superuser (Optional)**:
    - If you need an admin user, you can create one using the following command:
        ```bash
        python manage.py createsuperuser
        ```

3. **Run Development Server**:
    ```bash
    python manage.py runserver
    ```

4. **Access the Project**:
    - Open your web browser and go to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) to view the project.

5. **Access Admin Interface**:
    - If you created a superuser, you can access the admin interface at [http://127.0.0.1:8000/admin/](http://127.0.0.1:8000/admin/).

## Additional Notes

- Make sure to keep your `SECRET_KEY` secret, especially in production.
- Refer to the [Django documentation](https://docs.djangoproject.com/en/stable/) for more advanced configurations and features.

Now, you should be all set to start working with the Django project! If you encounter any issues or have questions, feel free to reach out or consult the Django documentation. Happy coding!
