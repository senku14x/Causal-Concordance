import splits as SP


def test_brew_split(brew_bank):
    sp = SP.brew_split(brew_bank)
    assert len(sp) == 100 and sum(v == "dev" for v in sp.values()) == 50
    assert all(sp[n] == "dev" for n in SP.BREW_FORCE_DEV)
    assert SP.brew_split(brew_bank) == sp                                   # deterministic
    by_z = {}
    for it in brew_bank:
        by_z.setdefault(it["intermediates"][0], []).append(sp[it["name"]])
    assert all(abs(v.count("dev") - v.count("test")) <= 2 for v in by_z.values())


def test_chain_split_groups_intact(chain_bank):
    items, groups = SP.chain_split(chain_bank)
    g = SP.chain_groups(chain_bank)
    assert len(g) == 20 and len(items) == 28
    for key, names in g.items():
        assert len({items[n] for n in names}) == 1                          # a step-sequence group never straddles
    assert items["chain-w2-d3p-0408"] == "dev"
    assert sum(v == "dev" for v in groups.values()) == 10
