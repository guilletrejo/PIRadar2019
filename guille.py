'''
for (34 alturas)
    

    for (todos los dias)

        #### TOODO EL BLOQUE SIGUIENTE ES PARA OBTENER EL DATO (12hs) DEL DIA ####
        
        Obtener primer dato (269, 269) y guardar en matriz_1
        Crear z_ex a partir de matriz_1 con axis = 0 ## z_ex tiene shape (1, 269, 269)
        for (12 horas)
            Obtener dato (269,269) y guardar en zaux
            Crear zaux_ex a partir de zaux con axis = 0 ## zaux_ex tiene shape (1, 269, 269)
            Concatena zaux_ex a z_ex en axis = 0        ## z_ex va a ir creciendo en shape (1 2 3 4 5 6..., 269, 269)
        
        ##########

        Obtener dato (12, 269, 269) y va a z_ex         ## (realizado en bloque anterior)
        Concatenar z_ex a una matriz vacia (1, 269, 269)## la matriz vacia se llama z_ex_all y se concatenan en axis = 0
        Entonces z_ex_all va a ir creciendo (12 24 36 ...., 269, 269)

        
        