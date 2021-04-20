### FUNKTIONEN ###

def last_x_month_sortiment(df, x_month, stichtag):    
    """
    Diese Funktion berechnet die Zeitfenster für Features mit Sortimentsgruppen.

    Parameters
    ----------
    df : Dataframe
        Long-Dataframe mit den Absätzen nach Produktgruppen.
    x_month : Integer
        Anzahl Monate, um wie viel zurück die Features berechnet werden sollen.
    stichtag : Date
        Stichtag, ab dem die Monate zurückgerechnet werden.

    Returns
    -------
    df_kk_grp_piv : Dataframe
        Dateframe mit erstellten Features pro Kunde für entsprechendes Zeitfenster.

    """
    import numpy as np
    
    ## DATUM   
    print("Stichtag:", stichtag)
    # Stichtag zu Year-Month
    stichmonth = np.datetime64(stichtag, "M")
    print("Stichmonat:", stichmonth)
    # x Monate subtrahieren
    print("Monate zurück:", x_month)
    start_dt = np.datetime64(stichmonth - (x_month -1), "D")
    print("Start-Monat:", start_dt)
    
    ## DATAFRAME
    # nach letzte x Monaten filtern
    df_kk_sel = df[(df["VERKAUFSDATUM"] <= stichtag) & (df["VERKAUFSDATUM"] >= start_dt)]     
    # Group-By und aggregieren
    df_kk_grp = df_kk_sel.groupby(["TKID", "SORTIMENTE"]).agg(
        ABSATZ_TOTAL = ("TKID", "size"))
    # long to wide
    df_kk_grp_piv = df_kk_grp.pivot_table(index = "TKID",
                                          columns = "SORTIMENTE",
                                          values = "ABSATZ_TOTAL")
    
    ## NAMING
    # Suffix für Features definieren
    name_neu = "_ABSATZ_" + str(x_month) + "M"
    print("Feature-Suffix: ", name_neu)
    # Suffix den Spaltennamen hinzufügen
    df_kk_grp_piv = df_kk_grp_piv.add_suffix(name_neu)
    # Tidy Feature-Names
    df_kk_grp_piv.columns = df_kk_grp_piv.columns.str.strip().str.upper().str.replace(' ', '_').str.replace('/', '_')
    
    ## INDEX
    # Reset Index       
    df_kk_grp_piv.reset_index(inplace=True)
    df_kk_grp_piv.rename_axis(None, axis=1, inplace=True)            
    print("Anz. Unique-Kunden:", df_kk_grp_piv["TKID"].nunique())
    
    # erstellte Features zurückgeben
    return df_kk_grp_piv

def set_zero_nan(df, x_month):    
    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    x_month : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    import numpy as np
    
    # Suffix erstellen
    name_suffix = "_" + str(x_month) + "M"
    print("Zeifenster:", name_suffix)
    # Variablennamen gemäss Suffix suchen
    filter_col = [col for col in df if col.endswith(name_suffix)]    
    
    # Wenn Kunde länger als x Monate oder gleich x Monate dabei, dann NaN-Werte mit 0 ersetzen 
    # -> Person war Kunde, hatte jedoch keinen Umsatz/Absatz im entsprechenden Zeitfenster
    df.loc[df["KUNDE_SEIT_MONATE"] >= x_month, filter_col] = df.loc[df["KUNDE_SEIT_MONATE"] >= x_month, filter_col].fillna(0)
    # Wenn Kunde kürzer als x Monate dabei, dann NaN-Werte setzen
    # -> Person war noch kein Kunde, konnte also keine Umsätze/Absätze generieren
    df.loc[df["KUNDE_SEIT_MONATE"] < x_month, filter_col] = np.NaN
    
    return df