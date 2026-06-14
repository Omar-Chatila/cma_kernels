def rotate_vector(a, b, c):
    import numpy as np

    d = np.array(a) - np.array(b)
    e = np.array(c) - np.array(b)

    angle_d = np.arctan2(d[1], d[0])
    theta = -angle_d

    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])

    return np.dot(rotation_matrix, e)


def calculate_steps_cor_grouped(
    dt_threshold,
    dt_tolerance,
    animal_trajectories,
    state_values=None,
):
    state_values = _resolve_state_values(animal_trajectories, state_values)
    steps = [[] for _ in state_values]
    state_to_index = {value: i for i, value in enumerate(state_values)}

    count_total = 0
    count_discarded = 0

    for _, entries in animal_trajectories.items():
        for i in range(2, len(entries)):
            count_total += 1

            time_diff_0 = (entries[i][2] - entries[i - 1][2]).total_seconds() / 60
            time_diff_1 = (entries[i - 1][2] - entries[i - 2][2]).total_seconds() / 60

            if (
                abs(time_diff_0 - dt_threshold) <= dt_threshold * dt_tolerance
                and abs(time_diff_1 - dt_threshold) <= dt_threshold * dt_tolerance
            ):
                state_index = _state_index(entries[i - 1][3], state_to_index, len(state_values))
                if state_index is None:
                    count_discarded += 1
                    continue

                a = (entries[i - 2][0], entries[i - 2][1])
                b = (entries[i - 1][0], entries[i - 1][1])
                c = (entries[i][0], entries[i][1])
                steps[state_index].append(rotate_vector(a, b, c))
            else:
                count_discarded += 1

    _print_step_counts(count_total, count_discarded, steps, state_values)
    return steps


def calculate_steps_brownian_grouped(
    dt_threshold,
    dt_tolerance,
    animal_trajectories,
    state_values=None,
):
    state_values = _resolve_state_values(animal_trajectories, state_values)
    steps = [[] for _ in state_values]
    state_to_index = {value: i for i, value in enumerate(state_values)}

    count_total = 0
    count_discarded = 0

    for _, entries in animal_trajectories.items():
        for i in range(1, len(entries)):
            count_total += 1
            time_diff = (entries[i][2] - entries[i - 1][2]).total_seconds() / 60

            if abs(time_diff - dt_threshold) <= dt_threshold * dt_tolerance:
                state_index = _state_index(entries[i - 1][3], state_to_index, len(state_values))
                if state_index is None:
                    count_discarded += 1
                    continue

                dx = entries[i][0] - entries[i - 1][0]
                dy = entries[i][1] - entries[i - 1][1]
                steps[state_index].append((dx, dy))
            else:
                count_discarded += 1

    _print_step_counts(count_total, count_discarded, steps, state_values)
    return steps


def _resolve_state_values(animal_trajectories, state_values):
    if state_values is not None:
        return list(state_values)

    values = set()
    for _, entries in animal_trajectories.items():
        for entry in entries:
            values.add(entry[3])
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=str)


def _state_index(raw_state, state_to_index, state_count):
    if raw_state in state_to_index:
        return state_to_index[raw_state]

    # Compatibility with old 1-based state trajectory tuples.
    try:
        state = int(raw_state) - 1
    except (TypeError, ValueError):
        return None
    return state if 0 <= state < state_count else None


def _print_step_counts(count_total, count_discarded, steps, state_values):
    discarded = (count_discarded / count_total) * 100.0 if count_total else 0.0
    print(f"  total of {count_total} steps, {discarded:.2f}% discarded")

    for idx, state_value in enumerate(state_values):
        print(f" State {state_value}: {len(steps[idx])}")
