from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = Path('config/opponent_features.yaml')


def _load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _rolling_sum(values, groups, window):
    return (
        values.groupby(groups)
        .rolling(window=window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )


def _season_order(df):
    return (
        df[['season', 'kickoff_time']]
        .groupby('season', as_index=False)['kickoff_time']
        .min()
        .sort_values('kickoff_time')['season']
        .tolist()
    )


def _team_presence(df):
    return (
        df[['season', 'team_code', 'team_name']]
        .dropna(subset=['team_code'])
        .drop_duplicates(['season', 'team_code'])
        .assign(team_code=lambda x: pd.to_numeric(x['team_code'], errors='coerce').astype('Int64'))
    )


def _add_pl_spell_id(base, source_df):
    seasons = _season_order(source_df)
    presence = _team_presence(source_df)
    teams_by_season = {
        s: set(presence.loc[presence['season'].eq(s), 'team_code'].dropna().astype(int))
        for s in seasons
    }

    rows = []
    all_codes = sorted(set().union(*teams_by_season.values()))
    for code in all_codes:
        spell = 0
        prev_present = False
        for season in seasons:
            present = code in teams_by_season[season]
            if not present:
                prev_present = False
                continue
            if not prev_present:
                spell += 1
            rows.append({'season': season, 'opponent_team_code': code, 'pl_spell_id': spell})
            prev_present = True

    spell_map = pd.DataFrame(rows)
    out = base.copy()
    out['opponent_team_code'] = pd.to_numeric(out['opponent_team_code'], errors='coerce').astype('Int64')
    out = out.merge(spell_map, on=['season', 'opponent_team_code'], how='left', validate='many_to_one')

    if out['pl_spell_id'].isna().any():
        bad = out.loc[out['pl_spell_id'].isna(), ['season', 'opponent_team_code', 'opponent_team_name']].drop_duplicates().head(20)
        raise ValueError('Could not assign PL spell id:\n' + bad.to_string(index=False))

    out['opponent_spell_key'] = (
        out['opponent_team_code'].astype(str)
        + '::spell'
        + out['pl_spell_id'].astype(int).astype(str)
    )
    return out


def _build_fixture_position_base(df, config):
    out = df.copy()
    out['kickoff_time'] = pd.to_datetime(out['kickoff_time'], utc=True, errors='coerce')
    out['position_group'] = out['position'].map(config['position_groups'])
    out['minutes'] = pd.to_numeric(out['minutes'], errors='coerce')
    points_col = config['points_column']
    out[points_col] = pd.to_numeric(out[points_col], errors='coerce')

    meaningful = out[out['minutes'] >= int(config['meaningful_minutes'])].copy()
    meaningful['opponent_team_code'] = pd.to_numeric(meaningful['opponent_team_code'], errors='coerce').astype('Int64')

    base = (
        meaningful.groupby(
            ['season', 'gameweek', 'fixture_id', 'kickoff_time', 'opponent_team_id',
             'opponent_team_code', 'opponent_team_name', 'position_group'],
            as_index=False,
            observed=True,
        )
        .agg(
            fixture_core_points_allowed=(points_col, 'sum'),
            fixture_meaningful_appearances=('player_code', 'size'),
            fixture_unique_players=('player_code', 'nunique'),
        )
    )
    base['fixture_avg_core_points_allowed'] = (
        base['fixture_core_points_allowed'] / base['fixture_meaningful_appearances']
    )
    return base


def _add_rolling_features(base, source_df, config):
    out = _add_pl_spell_id(base, source_df)
    out = out.sort_values(
        ['opponent_spell_key', 'position_group', 'kickoff_time', 'season', 'gameweek', 'fixture_id']
    ).reset_index(drop=True)

    group_key = out['opponent_spell_key'].astype(str) + '||' + out['position_group'].astype(str)
    grouped = out.groupby(['opponent_spell_key', 'position_group'], sort=False)
    shifted_points = grouped['fixture_core_points_allowed'].shift(1)
    shifted_apps = grouped['fixture_meaningful_appearances'].shift(1)
    out['opp_pos_prior_fixtures_spell'] = grouped.cumcount()

    for window in config['rolling_windows']:
        points = _rolling_sum(shifted_points.fillna(0), group_key, window)
        apps = _rolling_sum(shifted_apps.fillna(0), group_key, window)
        out[f'opp_pos_core_points_avg_l{window}'] = np.where(apps > 0, points / apps, np.nan)
        out[f'opp_pos_meaningful_apps_l{window}'] = apps

    return out


def _relegation_cohorts(source_df):
    seasons = _season_order(source_df)
    presence = _team_presence(source_df)
    teams = {
        s: set(presence.loc[presence['season'].eq(s), 'team_code'].dropna().astype(int))
        for s in seasons
    }
    cohorts = {}
    for i in range(1, len(seasons)):
        cohorts[seasons[i]] = teams[seasons[i-1]] - teams[seasons[i]]
    return seasons, cohorts


def _promoted_proxy_table(out, source_df, config):
    seasons, cohorts = _relegation_cohorts(source_df)
    season_idx = {s: i for i, s in enumerate(seasons)}
    n_years = int(config.get('promoted_proxy_relegation_seasons', 3))

    # final observation for each team-season-position group
    final_rows = (
        out.sort_values(['season', 'opponent_team_code', 'position_group', 'kickoff_time', 'fixture_id'])
        .groupby(['season', 'opponent_team_code', 'position_group'], as_index=False, observed=True)
        .tail(1)
    )

    rows = []
    for target_season in seasons:
        t_idx = season_idx[target_season]
        transition_seasons = [
            seasons[j]
            for j in range(max(1, t_idx - n_years + 1), t_idx + 1)
            if seasons[j] in cohorts
        ]

        relegated = set()
        for transition_season in transition_seasons:
            relegated |= cohorts[transition_season]
        if not relegated:
            continue

        candidate = final_rows[
            (final_rows['season'].map(season_idx) < t_idx)
            & final_rows['opponent_team_code'].astype('Int64').isin(relegated)
        ].copy()
        if candidate.empty:
            continue

        for pos_group, pos_df in candidate.groupby('position_group', observed=True):
            # latest available row per relegated club before target season
            per_club = (
                pos_df.sort_values(['opponent_team_code', 'kickoff_time'])
                .groupby('opponent_team_code', as_index=False)
                .tail(1)
            )
            row = {
                'season': target_season,
                'position_group': pos_group,
                'promoted_proxy_teams': per_club['opponent_team_code'].nunique(),
            }
            for window in config['rolling_windows']:
                col = f'opp_pos_core_points_avg_l{window}'
                row[f'promoted_proxy_core_points_avg_l{window}'] = per_club[col].mean()
            rows.append(row)

    return pd.DataFrame(rows)


def _add_fallbacks(out, source_df, config):
    proxy = _promoted_proxy_table(out, source_df, config)
    merged = out.merge(proxy, on=['season', 'position_group'], how='left', validate='many_to_one')

    for window in config['rolling_windows']:
        raw = f'opp_pos_core_points_avg_l{window}'
        proxy_col = f'promoted_proxy_core_points_avg_l{window}'
        filled = f'opp_pos_core_points_avg_l{window}_filled'
        source = f'opp_pos_core_points_avg_l{window}_source'

        merged[filled] = merged[raw].fillna(merged[proxy_col])
        merged[source] = np.select(
            [
                merged[raw].notna(),
                merged[raw].isna() & merged[proxy_col].notna(),
            ],
            ['team_history', 'promoted_proxy_3yr'],
            default='no_history',
        )

    return merged


def build_opponent_position_features(config_path=CONFIG_PATH, save=True):
    config = _load_config(config_path)
    input_path = Path(config['input_path'])
    output_path = Path(config['output_path'])
    df = pd.read_csv(input_path, low_memory=False)
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'], utc=True, errors='coerce')

    required = [
        'season', 'gameweek', 'fixture_id', 'kickoff_time', 'player_code', 'position',
        'minutes', 'core_total_points', 'team_code', 'team_name', 'opponent_team_id',
        'opponent_team_code', 'opponent_team_name'
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Historic fact table missing required columns: {missing}')

    base = _build_fixture_position_base(df, config)
    out = _add_rolling_features(base, df, config)
    out = _add_fallbacks(out, df, config)

    grain = ['season', 'fixture_id', 'opponent_team_code', 'position_group']
    dupes = out.duplicated(grain, keep=False)
    if dupes.any():
        sample = out.loc[dupes, grain + ['opponent_team_name']].head(20)
        raise ValueError('Duplicate opponent feature grain detected:\n' + sample.to_string(index=False))

    if save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
        print(f'Saved: {output_path}')
        print(f'Rows: {len(out):,}')
        print(f'Opponent teams: {out["opponent_team_code"].nunique():,}')
        print(f'Position groups: {sorted(out["position_group"].unique())}')
        print(f'Rolling windows: {config["rolling_windows"]}')
        print('\nFallback source counts:')
        for c in [c for c in out.columns if c.endswith('_source')]:
            print(f'{c}: {out[c].value_counts(dropna=False).to_dict()}')

    return out


def add_position_group_for_join(player_df, config_path=CONFIG_PATH):
    config = _load_config(config_path)
    out = player_df.copy()
    out['position_group'] = out['position'].map(config['position_groups'])
    return out


if __name__ == '__main__':
    build_opponent_position_features()
