# 200IQ Calculator<a href=#><img align="left" src="https://upload.wikimedia.org/wikipedia/commons/5/53/OpenCV_Logo_with_text.png"></a>


Desktop app cappable of analyse the ingame screen of League of Legends, and then send the data to a realtime database. Done with Python and the module OpenCV.

---


### Getting Started

> As the fist commit this repository is not ready to be fully functional, there are some issues.

Clone or download the repository, then follow the next instrucctions.

#### 1) Install Dependencies

- Install NPM packages with the package manager `npm install`.

#### 2) Issues

> As I can't publish the app in the AppStore yet, I can't fully share all of my code for security reasons.
> The only way to get this working is replasing those files, and that requires a firebase personal account.

1. The app needs the `google-services.json` wich I don't want to share in a public repository.
2. The app needs the `full credentials of the realtime database` wich I don't want to share in a public repository.

#### 3) Start the app

- Start the react native packager, run `npm start` from the root of your project.
- If you haven't already got an android device attached/emulator running then you'll need to get one running (make sure the emulator is with Google Play / APIs). When ready run `npm run android` or `yarn run android` from the root of your project.

If all has gone well you'll see an initial screen like the one below.
  
## Screenshot

![preview](https://i.imgur.com/CAsqXsc.png)

![preview](https://i.imgur.com/CunSMdF.png)

## Limitations

- Not recognizing buffs that reduces damage (eg. Annie's E, Warwick's E, Alistar's R, etc.).
- DPS not calculated correctly (on-hit effects and Critical Strike).
- LAN as only region to search.
- Calculations only working for Morgana and Karma.
- Calculations not using the following stats: Heal and Shield Power, Lethality, Armor Penetration, Lifesteal and Magic Penetration.
- Only tested on Android.
- No registration form yet.