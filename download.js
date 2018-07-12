
var data = source.data;
var filetext = file_text.data['start'][0];
// var filetext = 'label,id,SOFA,APACHEII,sex,event_type,vaso_type,vitalstatday28\n';
var splt = filetext.replace(/\n/g, '')
splt = splt.split(",");
for (i=0; i < data['index'].length; i++){ //for each event
    var currRow = [];
    for (x=0; x < splt.length; x++) { //for each column to download, concatenate
        currRow.push(data[splt[x]][i].toString());
        console.log(data[splt[x]][i].toString())
    }
    currRow[currRow.length] = currRow[currRow.length-1].toString().concat('\n'); //add new line to last entry
    var joined = currRow.join();
    filetext = filetext.concat(joined);
}

var filename = 'EVENTIVe_data.csv';
var blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' });

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename);
}

else {
    var link = document.createElement("a");
    link = document.createElement('a')
    link.href = URL.createObjectURL(blob);
    link.download = filename
    link.target = "_blank";
    link.style.visibility = 'hidden';
    link.dispatchEvent(new MouseEvent('click'))
}