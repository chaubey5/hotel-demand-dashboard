# Hotel Room Demand Forecasting Dashboard

A Streamlit dashboard based on the `cluster_code.ipynb` project.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL shown in the terminal (usually http://localhost:8501).
6. Upload `hotel_bookings.csv` in the sidebar.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and this README.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**, choose your repository and branch, and set the main file path to `app.py`.
4. Click **Deploy**.
5. When the app starts, open its public URL and upload `hotel_bookings.csv`.

## Dataset format

The CSV should include the columns used by the notebook, including:
`hotel`, `arrival_date_year`, `arrival_date_month`, `arrival_date_day_of_month`,
`is_canceled`, `stays_in_week_nights`, `stays_in_weekend_nights`, and
`reserved_room_type`.

The dashboard estimates demand from non-cancelled bookings. It does not measure actual occupied rooms.
"# hotel-demand-dashboard" 
