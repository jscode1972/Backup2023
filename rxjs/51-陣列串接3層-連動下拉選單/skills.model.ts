export interface Area {
    aid: string;
    area: string;
}

export interface Nation {
    nid: string;
    nation: string;
    aid: string;
}

export interface City {
    cid: string;
    city: string;
    nid: string;
}

export interface Full {
    cid: string;
    city: string;
    nid: string;
    nation: string;
    aid: string;
    area: string;
}

