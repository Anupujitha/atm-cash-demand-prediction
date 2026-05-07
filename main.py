# ATM Cash Demand Prediction & Route Optimization

# Install Libraries
!pip install xgboost ortools folium matplotlib

# Import Libraries
import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from math import radians, sin, cos, sqrt, atan2
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from google.colab import files

# Load Dataset
uploaded = files.upload()
df = pd.read_csv(list(uploaded.keys())[0])

# Select Features
df = df[['atmId','totalOutcome','numberOutcomeTransaction',
         'totalNumberTransaction','day']]

df = df.dropna()

# Encode Features
df['atmId'] = df['atmId'].astype('category').cat.codes
df['day'] = df['day'].astype('category').cat.codes

# Create Simulated Time Features
df['hour'] = np.random.randint(0, 24, len(df))
df['day_of_week'] = np.random.randint(0, 7, len(df))
df['month'] = np.random.randint(1, 13, len(df))

# Train Model
X = df.drop('totalOutcome', axis=1)
y = df['totalOutcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = XGBRegressor(
    n_estimators=100,
    max_depth=5
)

model.fit(X_train, y_train)

df['predicted'] = model.predict(X)

# Actual vs Predicted Graph
plt.figure(figsize=(10,5))
plt.plot(y_test.values[:100], label="Actual")
plt.plot(model.predict(X_test)[:100], label="Predicted")
plt.legend()
plt.title("Actual vs Predicted ATM Demand")
plt.show()

# Refill Decision
threshold = df['predicted'].mean()
df['Refill_Needed'] = df['predicted'] > threshold

refill_atms = df[df['Refill_Needed']]['atmId'].unique()
nodes = list(refill_atms)

print("ATMs needing refill:", nodes)

# Generate ATM Locations
base_lat, base_lon = 16.5, 81.7

atm_locations = {
    atm: (
        base_lat + np.random.uniform(-0.05, 0.05),
        base_lon + np.random.uniform(-0.05, 0.05)
    )
    for atm in df['atmId'].unique()
}

# Distance Function
def distance(coord1, coord2):
    R = 6371

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat/2)**2
        + cos(radians(lat1))
        * cos(radians(lat2))
        * sin(dlon/2)**2
    )

    a = min(1.0, max(0.0, a))

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

# Route Optimization
if len(nodes) > 1:

    nodes = nodes[:40]

    distance_matrix = []

    for i in nodes:
        row = []

        for j in nodes:
            row.append(
                int(
                    distance(
                        atm_locations[i],
                        atm_locations[j]
                    ) * 1000
                )
            )

        distance_matrix.append(row)

    manager = pywrapcp.RoutingIndexManager(
        len(nodes),
        1,
        0
    )

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return distance_matrix[
            manager.IndexToNode(from_index)
        ][
            manager.IndexToNode(to_index)
        ]

    transit_callback_index = routing.RegisterTransitCallback(
        distance_callback
    )

    routing.SetArcCostEvaluatorOfAllVehicles(
        transit_callback_index
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(search_parameters)

    route = []

    if solution:

        index = routing.Start(0)

        while not routing.IsEnd(index):

            route.append(
                nodes[
                    manager.IndexToNode(index)
                ]
            )

            index = solution.Value(
                routing.NextVar(index)
            )

        route.append(
            nodes[
                manager.IndexToNode(index)
            ]
        )

    print("Optimized Route:", route)

    # Map Visualization
    m = folium.Map(
        location=[base_lat, base_lon],
        zoom_start=11
    )

    atm_list_html = """
    <div style="
        position: fixed;
        top: 50px;
        left: 50px;
        width: 250px;
        height: 300px;
        overflow-y: auto;
        background-color: white;
        border: 2px solid black;
        padding: 10px;
        z-index:9999;
        font-size:14px;
    ">
    <b>ATMs needing refill:</b><br>
    """

    for atm in nodes:
        atm_list_html += f"• ATM {atm}<br>"

    atm_list_html += "</div>"

    m.get_root().html.add_child(
        folium.Element(atm_list_html)
    )

    # Plot ATM Locations
    for atm in nodes:

        lat, lon = atm_locations[atm]

        folium.CircleMarker(
            [lat, lon],
            radius=4,
            color='gray'
        ).add_to(m)

    # Plot Route
    for atm in route:

        lat, lon = atm_locations[atm]

        folium.Marker(
            [lat, lon],
            popup=f"ATM {atm}",
            icon=folium.Icon(color='red')
        ).add_to(m)

    route_coords = [
        atm_locations[i]
        for i in route
    ]

    folium.PolyLine(
        route_coords,
        color="blue",
        weight=4
    ).add_to(m)

    # Save Map
    m.save("atm_route_map.html")

    print("Map saved as atm_route_map.html")

else:
    print("Not enough ATMs for routing")

# Download Map
files.download("atm_route_map.html")