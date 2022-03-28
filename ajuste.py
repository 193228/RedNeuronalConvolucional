def ajusteDatos(ventana): #Obtenemos los datos de la vista
    valores = {}
    try:
        if ventana.radioEpocas.isChecked():
            epocas = 25
        else:
            epocas = int(ventana.epocas.text())

        if ventana.radioTaza.isChecked():
            eta = 0.01
        else:
            eta = float(ventana.inputGeneraciones.text())

        if ventana.radioLote.isChecked():
            lote = 32
        else:
            lote = int(ventana.lote.text())

        valores = {
            "epocas": epocas,
            "eta":eta,
            "lote": lote
        }

    except:
        print("datos incompletos")
        pass

    return valores