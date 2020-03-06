
/*
CREATE USER 'paracrawl_user'@'localhost' IDENTIFIED BY 'paracrawl_password';

CREATE DATABASE paracrawl CHARACTER SET 'utf8' COLLATE 'utf8_unicode_ci';
GRANT ALL PRIVILEGES ON paracrawl.* TO 'paracrawl_user'@'localhost';

mysql -u paracrawl_user -pparacrawl_password -Dparacrawl < create.sql
mysqldump -u paracrawl_user -pparacrawl_password --databases paracrawl | xz -c > db.xz
xzcat db.xz | mysql -u paracrawl_user -pparacrawl_password -Dparacrawl

sudo apt install python3-dev libpython3-dev python3-mysqldb
pip3 install mysql-connector-python
*/

DROP TABLE IF EXISTS document;
DROP TABLE IF EXISTS response;
DROP TABLE IF EXISTS url;
DROP TABLE IF EXISTS link;
DROP TABLE IF EXISTS url_align;
DROP TABLE IF EXISTS language;


CREATE TABLE IF NOT EXISTS url
(
    id INT AUTO_INCREMENT PRIMARY KEY,
    val TEXT,
    md5 VARCHAR(32) NOT NULL UNIQUE KEY
);

CREATE TABLE IF NOT EXISTS response
(
    id INT AUTO_INCREMENT PRIMARY KEY,
    url_id INT NOT NULL REFERENCES url(id),
    status_code INT,
    crawl_date DATETIME NOT NULL,
    to_url_id INT REFERENCES url(id),
    mime TINYTEXT,
    lang_id INT REFERENCES language(id),
    md5 VARCHAR(32)
);

CREATE TABLE IF NOT EXISTS link
(
    id INT AUTO_INCREMENT PRIMARY KEY,
    text TEXT,
    text_lang_id INT NOT NULL REFERENCES language(id),
    text_en TEXT,
    hover TEXT,
    image_url TEXT,
    document_id INT NOT NULL REFERENCES response(id),
    url_id INT NOT NULL REFERENCES url(id)
);

ALTER TABLE link
   ADD CONSTRAINT UQ_link_doc_url UNIQUE (document_id, url_id)
;

CREATE TABLE IF NOT EXISTS url_align
(
    id INT AUTO_INCREMENT PRIMARY KEY,
    url1 INT NOT NULL REFERENCES url(id),
    url2 INT NOT NULL REFERENCES url(id),
    score FLOAT
);

ALTER TABLE url_align
   ADD CONSTRAINT UQ_url_align_urls UNIQUE (url1, url2)
;

CREATE TABLE IF NOT EXISTS language
(
    id INT AUTO_INCREMENT PRIMARY KEY,
    lang VARCHAR(32)
);
