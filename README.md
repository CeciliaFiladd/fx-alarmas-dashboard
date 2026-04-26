# Sistema de alarmas FX — Streamlit Cloud deploy

Dashboard predictivo de tipo de cambio (USD vs ARS / BRL / CLP / COP / INR) que se actualiza con datos de Yahoo Finance en cada visita.

## Estructura

```
streamlit_app/
├── app.py                  # aplicación principal
├── requirements.txt        # dependencias
├── README.md               # este archivo
├── models/                 # modelos LightGBM persistidos (10 boosters)
│   ├── gbm_USD<CCY>_<H>.txt        # 10 clasificadores
│   ├── gbm_reg_USD<CCY>_<H>.txt    # 10 regresores
│   └── feature_columns.json
└── snapshots/              # datos que no se pueden bajar en vivo
    ├── fred_recent.json    # DFF, T10Y2Y, DGS10 últimos 80 días
    └── fx_recent.json      # backup FX si Yahoo falla
```

## Cómo funciona

En cada visita (con caché de 1 hora):

1. **Baja datos de Yahoo Finance** vía API JSON pública: 5 FX, DXY, VIX, ^TNX (10Y yield), Brent, WTI, cobre, oro, soja, hierro.
2. **Carga snapshot estático de FRED** para DFF (Fed Funds) y T10Y2Y (curva US 10Y-2Y) que no están en Yahoo.
3. **Computa 29 features** por moneda (mismas que en el notebook de entrenamiento).
4. **Carga 10 modelos LightGBM** (5 monedas × 2 horizontes 5d/20d) y hace inferencia.
5. **Renderiza el dashboard** con sparklines interactivos, niveles, alarmas reactivas y predicciones.

## Cómo deployar en Streamlit Cloud

### Paso 1 — Crear un repo público en GitHub

1. Andá a https://github.com/new
2. Ponele un nombre (ej. `fx-alarmas-dashboard`), marcá **Public**, dale "Create repository"
3. En tu computadora, abrí una terminal en la carpeta `streamlit_app/`:

```bash
cd "C:\Users\chech\Documents\Claude Code\Tipo de cambio\streamlit_app"
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<TU_USUARIO>/fx-alarmas-dashboard.git
git push -u origin main
```

(Si no tenés Git instalado, alternativa: subir los archivos manualmente desde GitHub web → "Add file" → "Upload files".)

### Paso 2 — Conectar a Streamlit Cloud

1. Andá a https://streamlit.io/cloud
2. Click en **"Sign in with GitHub"** (te logueás con tu cuenta de GitHub)
3. Click en **"Create app"** → **"Deploy a public app from GitHub"**
4. Seleccioná tu repo `fx-alarmas-dashboard`
5. **Branch**: `main`
6. **Main file path**: `app.py`
7. **App URL** (opcional): elegí un nombre — quedará `https://<nombre>.streamlit.app`
8. Click en **"Deploy"**

Tarda 2-3 minutos la primera vez (instala dependencias, levanta la app). Cuando termina, te queda la URL pública lista para compartir con tu equipo.

### Paso 3 — Compartir

La URL `https://<tu-nombre>.streamlit.app` es pública. Cualquiera con el link entra y ve el dashboard. **Cada visita ejecuta los scripts**, así que siempre ven datos frescos.

## Cómo actualizar los modelos

Los modelos LightGBM (`models/gbm_*.txt`) se entrenaron una vez con los scripts del proyecto principal (`scripts/15_modelo_produccion.py`). Si querés re-entrenarlos con datos más recientes:

1. Re-correr `scripts/15_modelo_produccion.py` en el proyecto principal
2. Copiar los `models/*.txt` actualizados a `streamlit_app/models/`
3. Hacer commit + push:
   ```bash
   git add models/
   git commit -m "Update models"
   git push
   ```
4. Streamlit Cloud detecta el push y redeployar automáticamente

## Cómo actualizar el snapshot de FRED

DFF y T10Y2Y cambian poco (semanas/meses). Cuando quieras refrescar el snapshot:

1. Re-correr el snippet de generación que está en `scripts/` del proyecto principal
2. Copiar `snapshots/fred_recent.json` actualizado
3. Commit + push

## Limitaciones conocidas

- **Yahoo Finance puede dar 429** (rate limit) en horas de mucho tráfico. La app cachea 1 hora para mitigar.
- **^TNX viene en formato ×10** a veces: la app lo divide automáticamente si detecta que la mediana es > 30.
- **^VIX, ^TNX y DX-Y.NYB** a veces tienen feed con delay; el último día puede aparecer como NaN. La app hace forward fill.

## Métricas del modelo (referencia)

- AUC clasificación a 5 días: ~0,57 promedio para BRL, CLP, COP. ARS no calibra bien clasificación pero tiene drift fuerte.
- RMSE regresión: NO le gana al random walk en los flotantes (consistente con Meese-Rogoff).
- Hit rate del signo: 86% para ARS a 20d con momentum/GBM.

Más detalle en los informes de Fase A/B/C del proyecto principal.
