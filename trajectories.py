from collections import Counter


def ensure_utm_geometry(gdf, geom_col: str = "geometry"):
    if {"utm_x", "utm_y"}.issubset(gdf.columns):
        import geopandas as gpd

        crs = None
        if "utm_crs" in gdf and gdf["utm_crs"].notna().any():
            crs = gdf["utm_crs"].dropna().iloc[0]
        return (
            gpd.GeoDataFrame(
                gdf.copy(),
                geometry=gpd.points_from_xy(gdf["utm_x"], gdf["utm_y"]),
                crs=crs,
            ),
            "geometry",
        )

    if geom_col not in gdf.columns:
        raise ValueError(f"Missing geometry column: {geom_col}")

    import geopandas as gpd

    df = gpd.GeoDataFrame(gdf.copy(), geometry=geom_col, crs=getattr(gdf, "crs", None))
    if df.crs is None:
        raise ValueError(
            "Kernel generation requires UTM coordinates. Provide utm_x/utm_y columns "
            "or a GeoDataFrame with a CRS so geographic coordinates can be projected."
        )
    if df.crs.is_geographic:
        utm_crs = df.estimate_utm_crs()
        if utm_crs is None:
            raise ValueError("Could not estimate a UTM CRS for kernel generation.")
        df = df.to_crs(utm_crs)
    elif not df.crs.is_projected:
        raise ValueError("Kernel generation requires projected UTM coordinates, not geographic coordinates.")
    return df, "geometry"


def infer_state_values(gdf, state_col: str = "state"):
    values = gdf[state_col].dropna().unique().tolist()
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=str)


def build_state_trajectories(
    gdf,
    *,
    id_col: str = "individual-local-identifier",
    time_col: str = "timestamp",
    geom_col: str = "geometry",
    state_col: str = "state",
):
    import pandas as pd

    gdf, geom_col = ensure_utm_geometry(gdf, geom_col)

    if state_col not in gdf.columns:
        raise ValueError(f"Missing state column: {state_col}")

    state_values = infer_state_values(gdf, state_col)
    animal_trajectories = {}

    df = gdf.copy()
    if time_col not in df.columns:
        df = df.reset_index()
    df[time_col] = pd.to_datetime(df[time_col])

    for animal_id, group in df.dropna(subset=[state_col]).groupby(id_col, sort=False):
        entries = []
        for _, row in group.sort_values(time_col).iterrows():
            geom = row[geom_col]
            entries.append((int(geom.x), int(geom.y), row[time_col], row[state_col]))
        animal_trajectories[animal_id] = entries

    all_states = []
    for _, entries in animal_trajectories.items():
        for row in entries:
            all_states.append(row[3])
    print(Counter(all_states))

    intervals = []
    for _, entries in animal_trajectories.items():
        val = detect_typical_interval(entries)
        if val:
            intervals.append(val)
    if not intervals:
        raise ValueError("Could not infer a typical interval from state trajectories.")

    import numpy as np

    dt_threshold = np.median(intervals)
    print("Erkannter Zeitabstand:", dt_threshold)
    return animal_trajectories, dt_threshold, state_values


def detect_typical_interval(entries):
    diffs = []
    for i in range(1, len(entries)):
        delta = ((entries[i][2] - entries[i - 1][2]).total_seconds() + 0.5) // 60
        diffs.append(delta)
    if len(diffs) == 0:
        return None
    return Counter(diffs).most_common(1)[0][0]
