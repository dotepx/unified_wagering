import os
import re
import pandas as pd
from pandas import DataFrame
from pandas import to_datetime
from utilities import JSONRequest, fetch_json
from dataclasses import dataclass
from datetime import date

@dataclass(frozen=True)
class MatchStats:
    """matchstats api config."""

    api_key: str = os.environ.get("MATCHSTAT_API_KEY", "").strip()
    host: str = "tennis-api-atp-wta-itf.p.rapidapi.com"
    url: str = "https://tennis-api-atp-wta-itf.p.rapidapi.com/tennis/v2"
    supplementals: tuple = (
        "round",
        "tournament",
        "tournament.tier",
        "tournament.court",
        "tournament.rank",
        "tournament.country",
        "tournament.site",
        "stat",
    )


def get_api(endpoint: str, params: dict) -> list:
    """one rate-limited matchstat api call"""
    return fetch_json(
        JSONRequest(
            url=f"{MatchStats.url}/{endpoint}",
            params=params,
            headers={
                "x-rapidapi-key": MatchStats.api_key,
                "x-rapidapi-host": MatchStats.host,
            },
        )
    )["data"]


def get_singles(division: str, top_n: int) -> DataFrame:
    """get top singles players data"""
    response: list = get_api(f"{division}/ranking/singles", {"pageSize": top_n})
    players = pd.DataFrame([x["player"] for x in response])
    players = players[["id", "name", "birthday", "countryAcr", "height"]].copy()
    players.columns = players.columns.str.upper()
    players["BIRTHDAY"] = pd.to_datetime(players["BIRTHDAY"])
    return players.astype({"ID": "int32"})


def get_matches(division: str, player_id: int) -> DataFrame:
    """completed match archive by player"""

    response: list = get_api(
        f"{division}/player/past-matches/{player_id}",
        {"include": ",".join(MatchStats.supplementals), "pageSize": 2000},
    )
    dataset = []
    for item in response:
        match_dict = {
            x: i
            for x, i in item.items()
            if x in ["id", "date", "player1Id", "player2Id", "match_winner", "result"]
        }
        # match_dict['date']=pd.to_datetime(match_dict['date'])
        results = match_dict["result"].split(" ")
        for idx, set_result in enumerate(results):
            scores = set_result.split("-")
            if len(scores) > 1:
                set_cnt: int = idx + 1
                match_dict[f"PLAYER1_SET{set_cnt}"] = scores[0]
                match_dict[f"PLAYER2_SET{set_cnt}"] = scores[1]

        match_dict["MATCHID"] = match_dict.pop("id")
        extended_dict = item["tournament"]
        if extended_dict:
            match_dict.update(
                {
                    x: i
                    for x, i in extended_dict.items()
                    if x in ["id", "name", "countryAcr"]
                }
            )
            match_dict["TOURNAMENTID"] = match_dict.pop("id")
            match_dict["TOURNAMENT"] = match_dict.pop("name")
            if extended_dict["court"]:

                match_dict.update(extended_dict["court"])
                match_dict["SURFACEID"] = match_dict.pop("id")
                match_dict["SURFACE"] = match_dict.pop("name")
            else:
                match_dict["SURFACEID"] = -1
            if extended_dict["rank"]:
                match_dict.update(extended_dict["rank"])
                match_dict["RANKID"] = match_dict.pop("id")
                match_dict["RANK"] = match_dict.pop("name")
            else:
                match_dict["RANKID"] = -1
            if item["round"]:
                match_dict.update(item["round"])
                match_dict["ROUNDID"] = match_dict.pop("id")
                match_dict["ROUND"] = match_dict.pop("name")
            else:
                match_dict["ROUNDID"] = item["roundId"]
        else:
            match_dict["TOURNAMENTID"] = item["tournamentId"]

        if "stats" in item:
            for player in ["player1", "player2"]:
                stats = item["stats"][player].items()
                if stats:
                    match_dict.update({f"{player}_{x}": i for x, i in stats})

        dataset.append(match_dict)
    df = pd.DataFrame(dataset)
    df.columns = df.columns.str.upper()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["DATE"] = df["DATE"].dt.normalize()
    df = df.astype(
        {x: "int32" for x in df.columns if x.endswith(("ID", "MATCH_WINNER"))}
    )
    df = df.astype({x: "float32" for x in df.columns if df[x].dtype == "float64"})
    return df


def get_fixtures(division: str,start_date:date,end_date:date):

    response: list = get_api(
        f"{division}/fixtures/{start_date.isoformat()}/{end_date.isoformat()}",
        {
            "include": ",".join(MatchStats.supplementals),
            "filter": "PlayerGroup:singles",
            "pageSize": 2000,
        },
    )
    print(response)


def main():
    fixturew=get_fixtures('wta',pd.to_datetime('2026.05.01'),pd.to_datetime('2026.06.01'))
    # wta_singles = get_singles("wta", 10)
    # id0 = wta_singles["ID"].values[0]
    # id1 = wta_singles["ID"].values[1]
    # wta_matches = pd.concat(
    #     [get_matches("wta", id0), get_matches("wta", id1)], ignore_index=True
    # )
    # print(wta_matches.info())
    # print(
    #     wta_matches[
    #         wta_matches["PLAYER1ID"].isin([id0, id1])
    #         & wta_matches["PLAYER2ID"].isin([id0, id1])
    #         # & (wta_matches["DATE"] == pd.to_datetime("2026-03-26"))
    #     ].T
    # )


if __name__ == "__main__":
    main()
